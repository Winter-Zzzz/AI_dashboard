import os
import sys
import torch
from tqdm import tqdm
import logging
import random
from datetime import datetime
import shutil
from transformers import EncoderDecoderCache

# 병렬성 경고 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 필요한 디렉토리 생성
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models', 'best_model'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from utils.data_loader import load_training_data
from utils.query_augmenter import QueryAugmenter 
from utils.query_augmenter_nlpaug import QueryAugmenterNlpAug
from config.model_config import ModelConfig

def backup_best_model():
    """기존 best_model 백업"""
    best_model_path = "models/best_model"
    if os.path.exists(best_model_path):
        backup_dir = "models/model_backups"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"best_model_{timestamp}")
        shutil.copytree(best_model_path, backup_path)
        logging.info(f"Previous best model backed up to: {backup_path}")
        shutil.rmtree(best_model_path)
    os.makedirs(best_model_path, exist_ok=True)

def train_model():
    # 로깅 설정
    log_file = os.path.join(project_root, 'logs', 'training_logs.txt')
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

    # 기존 모델 백업
    backup_best_model()
    
    # config 로드
    config = ModelConfig()
    
    # 데이터 로드
    data_file = os.path.join(project_root, 'data', 'raw', 'sample.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        logging.error("데이터셋이 비어있습니다!")
        return None, None

    # QueryAugmenter를 사용한 데이터 증강
    # augmenter = QueryAugmenter()
    augmenter = QueryAugmenterNlpAug()
    # 필요한 경우 추가 템플릿 등록
    # augmenter.add_src_dest_template("List {count} recent interactions between {src} and {dest}")
    # augmenter.add_src_dest_template("Get {count} newest messages from {src} to {dest}")
    
    aug_inputs, aug_outputs = augmenter.augment(input_texts, output_texts, num_variations=2)
    
    logging.info(f"원본 데이터셋 크기: {len(input_texts)}, 증강 후 크기: {len(aug_inputs)}")
    if len(aug_inputs) > len(input_texts):
        logging.info(f"샘플 증강 입력: {aug_inputs[len(input_texts)]}")  # 첫 번째 증강 데이터

    try:
        tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        logging.info("모델과 토크나이저 로드 완료")
    except Exception as e:
        logging.error(f"모델 로드 중 에러 발생: {str(e)}")
        return None, None

    optimizer = AdamW(model.parameters(), 
                     lr=config.LEARNING_RATE, 
                     weight_decay=config.WEIGHT_DECAY,
                     no_deprecation_warning=True)
    
    best_loss = float('inf')
    no_improve = 0
    dataset_size = len(aug_inputs)
    num_batches = (dataset_size + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    # past_key_values 초기화
    past_key_values = None  # 이 부분을 추가 (값이 없다면 None)

    try:
        for epoch in range(config.NUM_EPOCHS):
            # 매 에폭마다 데이터 셔플
            indices = list(range(dataset_size))
            random.shuffle(indices)
            shuffled_inputs = [aug_inputs[i] for i in indices]
            shuffled_outputs = [aug_outputs[i] for i in indices]

            model.train()
            total_loss = 0
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

            for i in progress_bar:
                start_idx = i * config.BATCH_SIZE
                end_idx = min((i + 1) * config.BATCH_SIZE, dataset_size)
                
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_outputs = shuffled_outputs[start_idx:end_idx]

                inputs = tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_LENGTH,
                    return_tensors="pt"
                )
                
                with torch.set_grad_enabled(True):
                    labels = tokenizer(
                        batch_outputs,
                        padding=True,
                        truncation=True,
                        max_length=config.MAX_LENGTH,
                        return_tensors="pt"
                    ).input_ids

                    if isinstance(past_key_values, tuple):
                        past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

                    outputs = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=labels,
                    )

                    loss = outputs.loss
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})


            avg_loss = total_loss / num_batches
            logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            print(f"\nEpoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                print("새로운 best model 저장!")
                save_path = 'models/best_model'
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logging.info(f"Best model 저장 (loss: {best_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    msg = f"\n{config.PATIENCE}회 동안 성능 향상이 없어 학습을 조기 종료합니다."
                    print(msg)
                    logging.info(msg)
                    break

            # 주기적인 테스트 생성
            if (epoch + 1) % 5 == 0:
                print("\n현재 모델 성능 테스트:")
                test_input = input_texts[0]  # 원본 데이터로 테스트
                print(f"입력: {test_input}")
                
                test_inputs = tokenizer(f"Generate JavaScript: {test_input}", 
                                     return_tensors="pt", 
                                     max_length=config.MAX_LENGTH, 
                                     padding=True, 
                                     truncation=True)
                
                with torch.no_grad():
                    test_outputs = model.generate(
                        input_ids=test_inputs['input_ids'],
                        attention_mask=test_inputs['attention_mask'],
                        max_length=config.MAX_GEN_LENGTH,
                        num_beams=config.NUM_BEAMS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        do_sample=True
                    )
                
                generated = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
                print(f"생성: {generated}\n")
                logging.info(f"테스트 생성 결과: {generated}")

    except Exception as e:
        error_msg = f"\n학습 중 에러 발생: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return None, None

    logging.info("학습 완료!")
    print("학습 완료!")
    return model, tokenizer

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    
    print("모델 학습을 시작합니다...")
    model, tokenizer = train_model()
    
    if model is not None and tokenizer is not None:
        print("학습이 성공적으로 완료되었습니다!")
    else:
        print("학습 실패!")