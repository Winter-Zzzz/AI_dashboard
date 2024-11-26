import os
import sys
import torch
from tqdm import tqdm
import logging
import random
from datetime import datetime
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 프로젝트 구조 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 필요한 디렉토리 생성
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models', 'best_model'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)

# 필요한 모듈 임포트
from src.utils.data_loader import load_training_data
from src.utils.query_augmenter_nlpaug import QueryAugmenterNlpAug
from src.config.model_config import ModelConfig

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

# 경고 메세지 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 데이터셋 클래스 정의
class QueryDataset(Dataset):
    """
        파이썬 코드 생성을 위한 커스텀 데이터셋
    """
    def __init__(self, input_texts, output_texts, tokenizer, max_length):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]


        # 입력 토큰화 시 특수 토큰 처리
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 출력 토큰화 시 파이썬 코드 구조 유지
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension added by tokenizer
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_model():
    """
        T5 모델을 사용하여 파이썬 코드 생성 모델을 학습하는 함수
    """
    # 로깅 설정
    log_file = os.path.join('logs', 'training_logs.txt')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)
    
    # 설정 및 데이터 로드
    config = ModelConfig()
    print(project_root)
    data_file = os.path.join(project_root, 'data', 'raw', 'function_sample.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        logging.error("Dataset is empty!")
        return None, None

    # 데이터 증강
    augmenter = QueryAugmenterNlpAug()
    aug_inputs, aug_outputs = augmenter.augment(input_texts, output_texts, num_variations=2)
    logging.info(f"Original dataset size: {len(input_texts)}, Size after augmentation: {len(aug_inputs)}")

    try:
        # 모델 및 토크나이저 초기화
        tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME, legacy=False)
        tokenizer.add_prefix_space = True
        model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    #     파이썬 코드에 특화된 특수 토큰 추가
    #     토크나이저 초기화 부분에서
        special_tokens = {
            'additional_special_tokens': [
                # 핵심 구조 토큰
                'print(',
                'TransactionFilter(data)',
                '.by_pk',
                '.by_src_pk',
                '.by_func_name',
                '.by_timestamp',
                '.sort',
                '.get_result()',
                'reverse=True',
                'reverse=False',
                
                # 기본 Python 토큰
                '(', ')', '[', ']', ':', ',', "'",
                
                # 메서드 체이닝
                '.'
              ]
      }


        # 특수 토큰 추가 및 토크나이저 업데이트
        tokenizer.add_special_tokens(special_tokens)

        # 모델의 임베딩 레이어 크기 조절 
        model.resize_token_embeddings(len(tokenizer))

        model = model.to(device)
        logging.info(f"Model and tokenizer loaded (Using {device})")

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

    # 데이터로더 설정
    train_dataset = QueryDataset(aug_inputs, aug_outputs, tokenizer, config.MAX_LENGTH)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # 옵티마이저 설정
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_loss = float('inf')
    no_improve = 0

    try:
        # 학습 루프
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

            for batch in progress_bar:
                # 데이터를 GPU/CPU로 이동
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 그래디언트 초기화
                optimizer.zero_grad()

                # 순전파
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # 최적화
                loss = outputs.loss
                total_loss += loss.item()

                # 역전파
                loss.backward()

                # 그래디언트 클래핑 (안정성 향상)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()

                progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            # 에포크 종료 후 처리
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                save_path = 'models/best_model'
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logging.info(f"Best model saved (loss: {best_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    msg = f"\nStopping early after {config.PATIENCE} epochs without improvement"
                    print(msg)
                    logging.info(msg)
                    break

            # 주기적 테스트 생성
            if (epoch + 1) % 3 == 0:
                model.eval()
                test_input = input_texts[0]
                print(f"\nTesting current model:")
                print(f"Input: {test_input}")
                
                # 파이썬 코드 생성을 위한 프롬프트
                test_inputs = tokenizer(
                    test_input,
                    return_tensors="pt",
                    max_length=config.MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    add_special_tokens=True
                ).to(device)

                # 향상된 생성 파라미터
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=test_inputs['input_ids'],
                        attention_mask=test_inputs['attention_mask'],
                        max_length=config.MAX_GEN_LENGTH,
                        num_beams=config.NUM_BEAMS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        do_sample=True,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                    )

                generated = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)
                print(f"Generated: {generated}\n")
                logging.info(f"Test generation result: {generated}")

    except Exception as e:
        error_msg = f"\nTraining error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return None, None

    logging.info("Training completed!")
    print("Training completed!")
    return model, tokenizer

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    print("Starting model training...")
    model, tokenizer = train_model()
    
    if model is not None and tokenizer is not None:
        print("Training completed successfully!")
    else:
        print("Training failed!")
