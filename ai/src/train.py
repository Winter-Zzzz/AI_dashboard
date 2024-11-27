import os
import sys
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

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

def train_model():
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
    
    config = ModelConfig()
    print(project_root)
    data_file = os.path.join(project_root, 'data', 'raw', 'generated_dataset.json')
    input_texts, output_texts = load_training_data(data_file)
    
    if len(input_texts) == 0 or len(output_texts) == 0:
        logging.error("Dataset is empty!")
        return None, None

    try:
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
                ')'
            ]
        }

        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        logging.info(f"Model and tokenizer loaded (Using {device})")

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

    train_dataset = QueryDataset(input_texts, output_texts, tokenizer, config.MAX_LENGTH)
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()

                progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                save_path = os.path.join(project_root, 'models', 'best_model')
                
                try:
                    temp_path = os.path.join(project_root, 'models', 'temp_model')
                    os.makedirs(temp_path, exist_ok=True)
                    
                    model.save_pretrained(temp_path)
                    tokenizer.save_pretrained(temp_path)
                    
                    if os.path.exists(save_path):
                        backup_path = f"{save_path}_backup"
                        if os.path.exists(backup_path):
                            shutil.rmtree(backup_path)
                        shutil.copytree(save_path, backup_path)
                    
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    shutil.copytree(temp_path, save_path)
                    
                    shutil.rmtree(temp_path)
                    
                    print(f"Model saved successfully to: {save_path}")
                    print(f"Saved files:", os.listdir(save_path))
                    logging.info(f"Best model saved (loss: {best_loss:.4f}) to {save_path}")
                    
                except Exception as e:
                    print(f"Error saving model: {str(e)}")
                    logging.error(f"Error saving model: {str(e)}")
                    if os.path.exists(f"{save_path}_backup"):
                        shutil.rmtree(save_path)
                        shutil.copytree(f"{save_path}_backup", save_path)
    
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    msg = f"\nStopping early after {config.PATIENCE} epochs without improvement"
                    print(msg)
                    logging.info(msg)
                    break

            if (epoch + 1) % 3 == 0:
                model.eval()
                test_input = input_texts[0]
                print(f"\nTesting current model:")
                print(f"Input: {test_input}")
                
                test_inputs = tokenizer(
                    test_input,
                    return_tensors="pt",
                    max_length=config.MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    add_special_tokens=True
                ).to(device)

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
                        bad_words_ids=[[tokenizer.pad_token_id]],
                    )

                def clean_generated_text(text: str) -> str:
                    special_tokens = ["<pad>", "</s>"]
                    for token in special_tokens:
                        text = text.replace(token, "")

                    text = " ".join(text.split())

                    text = text.replace(" (", "(")
                    text = text.replace(" )", ")")
                    text = text.replace(" .", ".")
                    text = text.replace(" [", "[")
                    text = text.replace(" ]", "]")
                    text = text.replace(" :", ":")
                    text = text.replace(" ,", ",")

                    return text.strip()

                generated = clean_generated_text(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True))
                
                print(f"Output: {generated}\n")
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