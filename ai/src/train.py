import os
import sys
import torch
from tqdm import tqdm
import random

from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
import re
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)

os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models', 'best_model'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)

from src.utils.data_loader import load_training_data
from src.config.model_config import ModelConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class QueryDataset(Dataset):
    def __init__(self, input_texts, output_texts, tokenizer, max_length):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.patterns = {
            'hex': re.compile(r'[0-9a-fA-F]{130}'),  # 130자리 16진수 값
            'time': re.compile(r'\b\d{10}\b'),
            'func': re.compile(r'\b\w+(?=\s+function\b)'),

        }

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
    
    def remove_special_tokens(text: str) -> str:
        """<pad>와 </s> 토큰 제거"""
        return text.replace('<pad>', '').replace('</s>', '').replace('<unk>', '')
    
    def remove_all_spaces(text: str) -> str:
        """모든 종류의 공백 문자 제거"""
        return ''.join(text.split())
    
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
        self.save_status()
        
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
        
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def save_status(self):
        status = {
            'current_epoch': self.training_history['epochs'][-1],
            'best_loss': self.best_loss,
            'last_train_loss': self.training_history['train_loss'][-1],
            'total_improvements': sum(self.training_history['improvement'])
        }
        
        status_file = os.path.join(self.log_dir, 'training_status.json')
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=4)

def train_model():
    config = ModelConfig()

    # 메모리 최적화
    torch.cuda.empty_cache()

    tracker = TrainingTracker(os.path.join(project_root, 'logs'))
    data_file = os.path.join(project_root, 'data', 'augmented', 'simplified_augmented_dataset.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        return None, None

    # 1. 토크나이저 설정
    # 토크나이저 먼저 생성
    tokenizer = T5Tokenizer.from_pretrained(
        config.MODEL_NAME,
        model_max_length=config.MAX_LENGTH,
        padding_side='right',
        truncation_side='right'
    )

    special_tokens = {
        'additional_special_tokens': [
            'to', 'from', 'by', 'all',
            'latest', 'oldest', 'earliest', 'recent', 'most recent',
            'after', 'before', 'between',
            # 기존 태그들
            '<hex>', '</hex>', 
            '<time>', '</time>',
            '<func>', '</func>',
        ]
    }
    
    # 특수 토큰 추가
    num_added_special = tokenizer.add_special_tokens(special_tokens)
    print(f"특수 토큰 {num_added_special}개 추가됨")

    # 4. 토큰 추가 검증
    print("\n토크나이저 어휘 체크:")
    test_sentence = "from latest transactions"
    tokens = tokenizer.encode(test_sentence, add_special_tokens=False)
    print(f"테스트 문장: {test_sentence}")
    print(f"토큰화 결과: {tokenizer.convert_ids_to_tokens(tokens)}")
    print(f"토큰 ID: {tokens}")

    tokenizer.pad_token = tokenizer.eos_token

 
    # 모델 초기화 및 임베딩 크기 조정
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # 3. 데이터셋 설정
    full_dataset = QueryDataset(input_texts, output_texts, tokenizer, config.MAX_LENGTH)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size 

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 4. 데이터로더 설정
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 5. 옵티마이저 설정
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, 
        eps=1e-8, 
    )
    
    # 6. 스케쥴러 설정
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    scaler = GradScaler()
    patience = config.PATIENCE
    no_improve = 0

    print("🚀 Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        torch.cuda.empty_cache()
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

            # 9. 진행 상황 표시 개선
            progress_bar.set_postfix({
                'loss': f'{loss.item() * config.ACCUMULATION_STEPS:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_train_loss / train_steps

        # 12. 검증
        model.eval()
        total_val_loss = 0
        val_steps = 0
        val_examples = []

        with torch.no_grad():
            val_batches = list(enumerate(val_dataloader))
            for batch_idx, batch in val_batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with autocast(device_type='cuda'):
                    output = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device),
                    )
                total_val_loss += output.loss.item()
                val_steps += 1

                # 검증 예시 저장
                if batch_idx % 10 == 0:  # validation 배치가 있는 경우에만
                    generated = model.generate(
                    input_ids=batch['input_ids'][0:1].to(device),
                    attention_mask=batch['attention_mask'][0:1].to(device),
                    max_length=config.MAX_GEN_LENGTH,
                    num_beams=config.NUM_BEAMS,
                    length_penalty=config.LENGTH_PENALTY,
                    no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                    early_stopping=config.EARLY_STOPPING
                    )
                    
                    val_examples.append({
                        'batch_idx': batch_idx,
                        'input': QueryDataset.normalize_spaces(QueryDataset.remove_special_tokens(
                            tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))),
                        'target': QueryDataset.remove_all_spaces(QueryDataset.remove_special_tokens(
                            tokenizer.decode(batch['labels'][0], skip_special_tokens=False))),
                        'output': QueryDataset.remove_all_spaces(QueryDataset.remove_special_tokens(
                            tokenizer.decode(generated[0], skip_special_tokens=False)))
                    })

        avg_val_loss = total_val_loss / val_steps
        
        # 13. 결과 출력 개선
        print(f"\nEpoch {epoch+1}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # 검증 예시 출력 - 배치 인덱스 포함
        print("\nValidation Examples:")
        for example in val_examples:
            print(f"\nBatch {example['batch_idx']}:")
            print(f"Input: {example['input']}")
            print(f"Target: {example['target']}")
            print(f"Output: {example['output']}")
        
        improved = tracker.update(epoch + 1, avg_val_loss)
        
        if improved:
            print(f"✨ New best loss achieved!")
            # 전체 state dict를 포함한 체크포인트 저장
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tokenizer_config': tokenizer.get_config_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
            }
            
            # 모델 가중치와 설정 저장
            output_dir = os.path.join(project_root, 'models', 'best_model')
            torch.save(checkpoint, os.path.join(output_dir, 'model_checkpoint.pt'))
            
            # 토크나이저와 모델 설정 저장
            model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"Saved checkpoint to {output_dir}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
            if no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break


    return model, tokenizer


if __name__ == "__main__":

   os.makedirs('logs', exist_ok=True)
   print("🚀 Starting model training...")
   model, tokenizer = train_model()
   
   if model is not None and tokenizer is not None:
       print("✅ Training completed successfully!")
   else:
       print("❌ Training failed!")

