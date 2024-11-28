import os
import sys
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
import re
import matplotlib.pyplot as plt
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

        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def show_training_example(model, batch, tokenizer, device, config):
    """학습 중인 배치에서 예시 출력을 생성하는 함수"""
    with torch.no_grad():
        generated = model.generate(
        input_ids=batch['input_ids'][:1].to(device),
        attention_mask=batch['attention_mask'][:1].to(device),
        max_length=config.MAX_GEN_LENGTH,
        num_beams=config.NUM_BEAMS,
        early_stopping=True,
)
        
        input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        target_text = tokenizer.decode(batch['labels'][0], skip_special_tokens=False)
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        
        print("\n=== Training Example ===")
        print(f"Input: {input_text}")
        print(f"Target: {target_text}")
        print(f"Generated: {generated_text}")
        print("-" * 80)

def evaluate_model(model, test_dataloader, tokenizer, device):
    """학습 완료 후 전체 테스트 데이터에 대한 평가를 수행하는 함수"""
    config=ModelConfig()
    model.eval()
    total_loss = 0
    all_examples = []
    
    print("\n=== Final Model Evaluation ===")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # 손실 계산
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            total_loss += outputs.loss.item()
            
            # 예시 생성
            generated = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_length=config.MAX_GEN_LENGTH,
                num_beams=config.NUM_BEAMS,
                early_stopping=True,
            )
            
            # 배치의 모든 예시 저장
            for i in range(len(generated)):
                input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                target_text = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
                generated_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                
                all_examples.append({
                    'input': input_text,
                    'target': target_text,
                    'generated': generated_text
                })
    
    avg_loss = total_loss / len(test_dataloader)
    print(f"\nAverage Test Loss: {avg_loss:.4f}")
    
    # 전체 결과를 파일로 저장
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        for i, example in enumerate(all_examples, 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"Input: {example['input']}\n")
            f.write(f"Target: {example['target']}\n")
            f.write(f"Generated: {example['generated']}\n")
            f.write("-" * 80 + "\n")
    
    return avg_loss, all_examples

class TrainingTracker:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
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
    tracker = TrainingTracker(os.path.join(project_root, 'logs'))
    data_file = os.path.join(project_root, 'data', 'augmented', 'augmented_dataset.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        return None, None

    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME, legacy=False)
    tokenizer.add_prefix_space = True
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    special_tokens = {
        'additional_special_tokens': [
            'print(TransactionFilter(data)',
            'get_result()',
            '.by_pk', 
            '.by_src_pk',
            '.by_timestamp',
            '.by_func_name',
            "('setup')",
            "('on')",
            "('off')",
            '.sort(reverse=True)',
            '.sort()',
            '.',
            ')',
            "by_func_name('off')",
            "by_func_name('on')", 
            "by_func_name('setup')",
            "print(TransactionFilter(data).by_func_name",
            "print(TransactionFilter(data).by_src_pk",
            "print(TransactionFilter(data).by_timestamp",
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Split dataset into train and validation sets (80:20)
    
    full_dataset = QueryDataset(input_texts, output_texts, tokenizer, config.MAX_LENGTH)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size  # This ensures the sizes sum to the total

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), 
                     lr=config.LEARNING_RATE,
                     weight_decay=config.WEIGHT_DECAY, 
                     eps=1e-8, 
                     betas=(0.9, 0.999))
    
    # 워밍업이 포함된 코사인 스케줄러로 변경
    num_training_steps = len(train_dataloader) * config.NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    patience = config.PATIENCE
    no_improve = 0
    best_loss = float('inf')
    accumulation_steps = 4

    print("🚀 Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            with torch.cuda.amp.autocast():
                output = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device), 
                    labels=batch['labels'].to(device)
                )
            
            loss = output.loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            if (i + 1) % 50 == 0:
                show_training_example(model, batch, tokenizer, device, config)

        # Handle last batch if needed
        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                total_val_loss += output.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        improved = tracker.update(epoch + 1, avg_val_loss)
        
        if improved:
            print(f"✨ New best loss achieved!")
            model.save_pretrained(os.path.join(project_root, 'models', 'best_model'))
            tokenizer.save_pretrained(os.path.join(project_root, 'models', 'best_model'))
            no_improve = 0
            best_loss = avg_val_loss
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