import os
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from datetime import datetime
from torch.amp import GradScaler, autocast
import json
import sys

import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(project_root, 'ai', 'src'))  # ai/src 경로 추가

# ai 디렉토리 내부의 logs 및 models 폴더 생성
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)


from config.fine_tuning_config import ModelConfig
from utils.data_loader import load_training_data

class QueryDataset(Dataset):
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


class TrainingTracker:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.status_file = os.path.join(self.log_dir, 'training_status.json')

        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                self.best_loss = status.get('best_loss', float('inf'))
        else:
            self.best_loss = float('inf')
            
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'improvement': [],
        }
        self.best_loss = float('inf')
    
    def update(self, epoch, train_loss):
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        
        improved = train_loss < self.best_loss
        self.training_history['improvement'].append(improved)
        if improved:
            self.best_loss = train_loss
        
        self.plot_progress()
        self.save_status(epoch)
        
        return improved
    
    def plot_progress(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_history['epochs'], 
                self.training_history['train_loss'], 
                label='Training Loss', 
                marker='o')
        
        improved_epochs = [e for i, e in enumerate(self.training_history['epochs']) 
                         if self.training_history['improvement'][i]]
        improved_losses = [l for i, l in enumerate(self.training_history['train_loss']) 
                         if self.training_history['improvement'][i]]
        
        if improved_epochs:
            plt.scatter(improved_epochs, improved_losses, 
                       color='green', s=100, 
                       label='Improvement', 
                       zorder=5)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # 현재 시간 추가하여 훈련 진행 그래프 파일명 설정
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_progress_file = f'training_progress_{current_time}.png'        
        plt.savefig(epoch_progress_file)
        plt.close()
    
    def save_status(self, epoch):
        status = {
            'current_epoch': self.training_history['epochs'][-1],
            'best_loss': self.best_loss,
            'last_train_loss': self.training_history['train_loss'][-1],
            'total_improvements': sum(self.training_history['improvement'])
        }
        
        # 에포크 번호를 파일명에 추가하여 훈련 상태 저장
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_status_file = os.path.join(self.log_dir, f'training_status_epoch_{epoch}_{current_time}.json')
        status_dir = os.path.dirname(epoch_status_file)
        if status_dir:  # Only create directory if path contains a directory
            os.makedirs(status_dir, exist_ok=True)

        with open(epoch_status_file, 'w') as f:
            json.dump(status, f, indent=4)

def fine_tune_model():
    config = ModelConfig()  # 이 부분을 먼저 선언
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(project_root, 'models', 'fine_tuned_model')
    
    # 모델 디렉토리 및 checkpoint 경로 설정
    model_dir = os.path.join(project_root, 'models', 'best_model')
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pt")
    
    # 경로 존재 확인
    if not os.path.exists(model_path):
        print(f"모델 디렉토리가 존재하지 않습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 학습해주세요.")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
        print("먼저 train.py를 실행하여 모델을 학습해주세요.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    print(f"Loaded tokenizer from {model_dir}")

    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
        
    print(f"학습된 모델 로딩 완료! (Device: {device})")

    # 데이터셋 로드
    data_file = os.path.join(project_root, 'data', 'augmented_dataset.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        return None, None

    # 데이터셋 설정
    train_dataset = QueryDataset(input_texts, output_texts, tokenizer, config.MAX_LENGTH)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # optimizer 초기화 및 상태 로드
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, 
        eps=1e-8
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # scheduler 초기화 및 상태 로드
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # scaler 초기화 및 상태 로드
    scaler = GradScaler()
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 훈련 상태 추적기 초기화
    tracker = TrainingTracker(os.path.join(project_root, 'logs'))
    
    # early stopping 관련 변수 초기화
    patience = config.PATIENCE
    no_improve = 0
    start_epoch = checkpoint['epoch'] + 1
    best_loss = float('inf')

    print("🚀 Resuming fine-tuning from epoch", start_epoch)
    
    # 여기서부터 training loop 시작 (기존 epoch 루프를 start_epoch부터 시작)
    for epoch in range(start_epoch, start_epoch + config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            with autocast(device_type='cuda'):
                output = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device), 
                    labels=batch['labels'].to(device)
                )
            
                loss = output.loss / config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()

            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)  # 그래디언트 클리핑 전에 스케일링 해제
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                scaler.step(optimizer)  # Optimizer 업데이트
                scaler.update()  # 스케일러 업데이트
                scheduler.step()  # 스케줄러 업데이트
                optimizer.zero_grad()

            # 실제 손실값 저장
            total_train_loss += loss.item() * config.ACCUMULATION_STEPS
            train_steps += 1

            # 진행 상황 표시
            progress_bar.set_postfix({
                'loss': f'{loss.item() * config.ACCUMULATION_STEPS:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_train_loss / train_steps

        print(f"\nEpoch {epoch+1}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        print("\n=== Test Generation ===")
        model.eval()
        test_texts = [
            "Load second recent txn",
            "Get second recent transaction",
            "Show second recent result"
        ]

        for test_text in test_texts:
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=100,
                    num_beams=10,  # 증가
                    length_penalty=0.6,  # 추가
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    bad_words_ids=[[tokenizer.encode(word, add_special_tokens=False)[0]] for word in ['-src', 'pital']]  # 잘못된 토큰 방지
                )
            
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input: {test_text}")
            print(f"Output: {predicted_text}\n")

        # 트래커에 저장
        improved = tracker.update(epoch + 1, avg_train_loss)
        
        if improved:
            print(f"✨ New best loss achieved!")
            # 모델 저장
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.get_vocab(),  # 어휘 저장
                'tokenizer_special_tokens_map': tokenizer.special_tokens_map,  # 특수 토큰 맵 저장
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'loss': avg_train_loss,
            }

            output_dir = os.path.join(project_root, 'models', 'fine_tuned_model')
            os.makedirs(output_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(output_dir, 'model_checkpoint.pt'))
            
            model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"Saved checkpoint to {output_dir}")
            tracker.save_status(epoch+1)
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
            if no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break


    return model, tokenizer 


if __name__ == "__main__":
    print("🚀 Starting fine-tuning process...")
    model, tokenizer = fine_tune_model()

    if model is not None and tokenizer is not None:
        print("✅ Fine-tuning completed successfully!")
    else:
        print("❌ Fine-tuning failed!")
