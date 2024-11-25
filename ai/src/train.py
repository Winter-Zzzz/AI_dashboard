import os
import sys
import torch
from tqdm import tqdm
import logging
import random
from datetime import datetime
import shutil
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 필요한 디렉토리 생성
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models', 'best_model'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)

# Utils import
from src.utils.data_loader import load_training_data
# from src.utils.query_augmenter import QueryAugmenter 
from src.utils.query_augmenter_nlpaug import QueryAugmenterNlpAug
from src.config.model_config import ModelConfig

# GPU setup remains the same
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

# Disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom dataset class
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

        # Tokenize inputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
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
    # Logging setup remains the same
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
    
    # Load config and data
    config = ModelConfig()
    data_file = os.path.join(project_root, 'data', 'raw', 'function_sample.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        logging.error("Dataset is empty!")
        return None, None

    # Data augmentation remains the same
    augmenter = QueryAugmenterNlpAug()
    aug_inputs, aug_outputs = augmenter.augment(input_texts, output_texts, num_variations=2)
    logging.info(f"Original dataset size: {len(input_texts)}, Size after augmentation: {len(aug_inputs)}")

    try:
        tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        model = model.to(device)
        logging.info(f"Model and tokenizer loaded (Using {device})")
        logging.info(f"Model and tokenizer loaded (Using {device})")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

    # Create dataset and dataloader
    train_dataset = QueryDataset(aug_inputs, aug_outputs, tokenizer, config.MAX_LENGTH)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_loss = float('inf')
    no_improve = 0

    try:
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            # Save best model
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

            # Periodic test generation
            if (epoch + 1) % 5 == 0:
                model.eval()
                test_input = input_texts[0]
                print(f"\nTesting current model:")
                print(f"Input: {test_input}")
                
                test_inputs = tokenizer(
                    f"Generate JavaScript: {test_input}",
                    return_tensors="pt",
                    max_length=config.MAX_LENGTH,
                    padding=True,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=test_inputs['input_ids'],
                        attention_mask=test_inputs['attention_mask'],
                        max_length=config.MAX_GEN_LENGTH,
                        num_beams=config.NUM_BEAMS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        do_sample=True
                    )

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
