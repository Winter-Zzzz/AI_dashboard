from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
from torch.cuda.amp import autocast
import torch
import json
import logging
import re
import os
from typing import Tuple
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.config.fine_tuning_config import ModelConfig

class QueryDataset(Dataset):
    def __init__(self, input_texts, output_texts, tokenizer, max_length=None):
        self.config = ModelConfig()
        self.tokenizer = tokenizer
        self.max_length = max_length or self.config.MAX_INPUT_LENGTH
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.patterns = {name: re.compile(pattern) for name, pattern in self.config.PATTERNS.items()}

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        # 각 아이템마다 개별적으로 토크나이징 수행
        input_text = self.process_text(self.input_texts[idx])
        output_text = self.output_texts[idx]

        # 입력 텍스트 토크나이징
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 출력 텍스트 토크나이징 - 특수 문자 유지
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': input_encoding.input_ids.squeeze(0),
            'attention_mask': input_encoding.attention_mask.squeeze(0),
            'labels': target_encoding.input_ids.squeeze(0)
        }

    def process_text(self, text: str) -> str:
        """텍스트 전처리 - 단어 간 공백 정규화 및 패턴 매칭"""
        text = ' '.join(text.split())
        
        matches_info = []
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches_info.append({
                    'start': match.start(),
                    'end': match.end(),
                    'pattern_name': pattern_name,
                    'matched_text': match.group().strip()
                })
        
        matches_info.sort(key=lambda x: x['start'], reverse=True)
        
        for match_info in matches_info:
            tag_start = f"<{match_info['pattern_name']}>"
            tag_end = f"</{match_info['pattern_name']}>"
            tagged_text = f"{tag_start}{match_info['matched_text']}{tag_end}"
            tagged_text = ''.join(tagged_text.split())
            
            text = (
                text[:match_info['start']] +
                tagged_text +
                text[match_info['end']:]
            )
        
        return text

    @staticmethod
    def remove_special_tokens(text: str) -> str:
        """특수 토큰 제거"""
        return text.replace('<pad>', '').replace('</s>', '').replace('<unk>', '')

    @staticmethod
    def remove_all_spaces(text: str) -> str:
        """모든 공백 제거"""
        return ''.join(text.split())

class QueryParserFineTuner:
    def __init__(self, model_path: str = None):
        self.config = ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        try:
            if self.config.USE_BASE_MODEL and os.path.exists(self.config.BASE_MODEL_PATH):
                model_path = self.config.BASE_MODEL_PATH
                logging.info(f"Loading trained model from {model_path}")
            else:
                model_path = model_path or self.config.MODEL_NAME
                logging.info(f"Loading base model: {model_path}")
            
            # 토크나이저와 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            # 현재 특수 토큰 확인
            logging.info("Current special tokens:")
            logging.info(self.tokenizer.special_tokens_map)
            logging.info(f"Vocabulary size: {len(self.tokenizer)}")
            
            # Fine-tuning을 위한 레이어 선택적 동결
            self.freeze_base_layers()

            self.model = self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
        
        torch.cuda.empty_cache()

    def freeze_base_layers(self):
        """기본 레이어는 동결하고 마지막 몇 개 레이어만 학습하도록 설정"""
        encoder_layers = self.model.encoder.block
        num_layers = len(encoder_layers)
        layers_to_freeze = int(0.75 * num_layers)
        
        for i in range(layers_to_freeze):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
                
        logging.info(f"Frozen {layers_to_freeze} encoder layers out of {num_layers}")

    def validate_tokens(self, sample_text: str):
        """토큰화 결과 검증"""
        tokens = self.tokenizer.tokenize(sample_text)
        logging.info(f"\nSample text: {sample_text}")
        logging.info(f"Tokenized: {tokens}")
        return tokens

    def prepare_dataset(self, data_path: str = None, eval_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """데이터셋 준비 및 학습/검증 분할"""
        data_path = data_path or self.config.DATA_PATH
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info("Sample data format:")
        for i, item in enumerate(data['dataset'][:3]):
            logging.info(f"Input {i}: {item['input']}")
            logging.info(f"Output {i}: {item['output']}\n")
            
        texts = [item['input'] for item in data['dataset']]
        labels = [item['output'] for item in data['dataset']]
        
        full_dataset = QueryDataset(texts, labels, self.tokenizer, max_length=self.config.MAX_INPUT_LENGTH)
        
        dataset_size = len(full_dataset)
        eval_size = int(dataset_size * eval_split)
        train_size = dataset_size - eval_size
        
        train_dataset, eval_dataset = random_split(
            full_dataset, 
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logging.info(f"Training set size: {len(train_dataset)}")
        logging.info(f"Evaluation set size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset

    def train(self, train_dataset, eval_dataset):
        """모델 파인튜닝"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.OUTPUT_DIR, f"checkpoint_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        eval_steps = max(len(train_dataset) // (self.config.BATCH_SIZE * 5), 1)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            warmup_ratio=self.config.WARMUP_RATIO,
            weight_decay=self.config.WEIGHT_DECAY,
            logging_dir=self.config.LOGGING_DIR,
            logging_steps=self.config.LOGGING_STEPS,
            save_strategy=self.config.SAVE_STRATEGY,
            save_steps=eval_steps,
            eval_strategy=self.config.EVAL_STRATEGY,
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=self.config.ACCUMULATION_STEPS,
            max_grad_norm=self.config.GRADIENT_CLIP,
            save_total_limit=1,
            report_to="none",
            fp16=self.config.FP16,
            dataloader_num_workers=self.config.NUM_WORKERS,
            dataloader_pin_memory=self.config.PIN_MEMORY,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        try:
            logging.info("Starting fine-tuning...")
            
            # Trainer에 의해 관리되는 단일 학습 호출
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # 최종 평가
            eval_metrics = trainer.evaluate()
            
            # 결과 로깅
            logging.info("Training completed!")
            logging.info(f"Final training metrics: {metrics}")
            logging.info(f"Final evaluation metrics: {eval_metrics}")
            
            # 최종 모델 저장
            trainer.save_model(os.path.join(output_dir, 'final_model'))
            
            # 샘플 예측으로 결과 확인
            self.test_predictions(eval_dataset, n_samples=3)
            
        except Exception as e:
            logging.error(f"Fine-tuning failed: {str(e)}")
            raise


    def test_predictions(self, eval_dataset, n_samples=3):
        """샘플 데이터로 예측 테스트"""
        indices = torch.randperm(len(eval_dataset))[:n_samples]
        
        logging.info("\nTest Predictions:")
        for idx in indices:
            example = eval_dataset[idx]
            input_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            target_text = self.tokenizer.decode(example['labels'], skip_special_tokens=True)
            
            # 예측 수행
            predicted_text = self.predict(input_text)
            
            logging.info(f"\nInput: {input_text}")
            logging.info(f"Target: {target_text}")
            logging.info(f"Predicted: {predicted_text}")

    def predict(self, query: str) -> str:
        """최적화된 예측 함수"""
        self.model.eval()
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_INPUT_LENGTH
        ).to(self.device)

        with torch.no_grad(), autocast(enabled=True):
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.MAX_LENGTH,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tuner = QueryParserFineTuner()
    train_dataset, eval_dataset = tuner.prepare_dataset()
    tuner.train(train_dataset, eval_dataset)
    
if __name__ == "__main__":
    main()