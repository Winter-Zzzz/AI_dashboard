from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
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
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length or self.config.MAX_INPUT_LENGTH
        self.patterns = {name: re.compile(pattern) for name, pattern in self.config.PATTERNS.items()}

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.process_text(self.input_texts[idx])
        output_text = self.output_texts[idx]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

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
        """<pad>와 </s> 토큰 제거"""
        return text.replace('<pad>', '').replace('</s>', '').replace('<unk>', '')
    
    @staticmethod
    def remove_all_spaces(text: str) -> str:
        """모든 종류의 공백 문자 제거"""
        return ''.join(text.split())
    
    @staticmethod
    def normalize_spaces(text: str) -> str:
        """연속된 공백을 하나의 공백으로 반환"""
        return ' '.join(text.split())


class QueryParserFineTuner:
    def __init__(self, model_path: str = None):
        self.config = ModelConfig()
        
        try:
            # 모델 경로 결정
            if self.config.USE_BASE_MODEL and os.path.exists(self.config.BASE_MODEL_PATH):
                model_path = self.config.BASE_MODEL_PATH
                logging.info(f"Loading trained model from {model_path}")
            else:
                model_path = model_path or self.config.MODEL_NAME
                logging.info(f"Loading base model: {model_path}")
            
            # 모델과 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            # 특수 토큰 추가
            special_tokens = {
                'additional_special_tokens': [
                    '<hex>', '</hex>', 
                    '<time>', '</time>',
                    '<func>', '</func>',
                ]
            }
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            if num_added_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"Model loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
        
        torch.cuda.empty_cache()

    def prepare_dataset(self, data_path: str = None, eval_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """데이터셋 준비 및 학습/검증 분할"""
        data_path = data_path or self.config.DATA_PATH
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        texts = [item['input'] for item in data['dataset']]
        labels = [item['output'] for item in data['dataset']]
        
        full_dataset = QueryDataset(texts, labels, self.tokenizer, max_length=self.config.MAX_INPUT_LENGTH)
        
        # 데이터셋을 학습용과 검증용으로 분할
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
        
        return train_dataset, eval_dataset  # 이 부분이 누락되어 있었습니다

    def train(self, train_dataset, eval_dataset):
        """모델 파인튜닝"""
        output_dir = self.config.OUTPUT_DIR
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,  # 평가용 배치 사이즈 추가
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            logging_dir=self.config.LOGGING_DIR,
            logging_steps=self.config.LOGGING_STEPS,
            save_strategy=self.config.SAVE_STRATEGY,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy=self.config.EVAL_STRATEGY, 
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # 최적 모델 선택 기준
            greater_is_better=False,  # loss는 낮을수록 좋음
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # 평가용 데이터셋 추가
            tokenizer=self.tokenizer,
        )

        try:
            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")
            
            # 최종 평가 수행
            eval_results = trainer.evaluate()
            logging.info(f"Final evaluation results: {eval_results}")
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, query: str) -> str:
        """파인튜닝된 모델로 예측"""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_INPUT_LENGTH
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                **self.config.GENERATION_CONFIG
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tuner = QueryParserFineTuner()
    train_dataset, eval_dataset = tuner.prepare_dataset()
    tuner.train(train_dataset, eval_dataset)
    
    test_query = "show most recent get function transaction"
    result = tuner.predict(test_query)
    print(f"Query: {test_query}")
    print(f"Generated code: {result}")

if __name__ == "__main__":
    main()