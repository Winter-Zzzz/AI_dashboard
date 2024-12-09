import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import sys
import os
from typing import Dict, List, Optional
import logging
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenLLaMA_Augmenter:
    def __init__(self, model_name: str = "openlm-research/open_llama_3b", device: Optional[str] = None):
        logging.info("모델과 토크나이저를 로드중...")
        
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            # 토크나이저 초기화 및 패딩 토큰 설정
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                legacy=True,
                trust_remote_code=True,
                padding_side='left'
            )
            # 패딩 토큰이 없는 경우 eos 토큰을 패딩 토큰으로 사용
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            
            # 모델의 패딩 토큰 ID 설정
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            logging.info(f"모델이 성공적으로 로드되었습니다. 사용 디바이스: {self.device}")
            
        except Exception as e:
            logging.error(f"모델 초기화 중 오류 발생: {str(e)}")
            raise

    def create_prompt(self, input_query: str, output_query: str) -> str:
        return f"""트랜잭션 쿼리 생성기로서 주어진 쿼리의 논리적 의미를 유지하면서 변형된 쿼리를 생성하세요.

원본 입력: {input_query}
원본 출력: {output_query}

새로운 입력 쿼리는 동일한 출력에 매핑되어야 합니다. 새 쿼리는 의미적으로 동일하지만 다른 단어, 동의어 또는 구조를 사용해야 합니다.

새로운 입력:"""

    def generate_batch_variations(self, prompts: List[str], batch_size: int = 4) -> List[str]:
        """배치 단위로 쿼리 변형을 생성합니다."""
        try:
            # 토크나이즈 및 패딩
            tokenized = [self.tokenizer(p, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0) 
                        for p in prompts]
            
            # 패딩 적용
            max_len = max(len(t) for t in tokenized)
            padded_inputs = []
            attention_masks = []
            
            for tokens in tokenized:
                padding_length = max_len - len(tokens)
                padded = torch.cat([
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=tokens.dtype),  # 패딩을 앞에 추가
                    tokens
                ])
                padded_inputs.append(padded)
                attention_masks.append(torch.cat([
                    torch.zeros(padding_length),  # attention mask도 앞에 0을 추가
                    torch.ones(len(tokens))
                ]))
            # 배치로 변환
            input_ids = torch.stack(padded_inputs).to(self.device)
            attention_mask = torch.stack(attention_masks).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    temperature=0.75,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
        except Exception as e:
            logging.error(f"배치 생성 중 오류 발생: {str(e)}")
            return [""] * len(prompts)

    def augment_dataset(self, dataset: Dict, num_variations: int = 1, batch_size: int = 4) -> Dict:
        augmented_data = []
        failed_generations = 0
        
        # 원본 데이터 포함
        augmented_data.extend(dataset['dataset'])
        
        logging.info(f"각 쿼리당 {num_variations}개의 변형을 생성중... (배치 크기: {batch_size})")
        
        # 배치 처리를 위한 준비
        current_batch = []
        current_items = []
        
        for item in tqdm(dataset['dataset']):
            for _ in range(num_variations):
                prompt = self.create_prompt(item['input'], item['output'])
                current_batch.append(prompt)
                current_items.append(item)
                
                if len(current_batch) >= batch_size:
                    try:
                        new_inputs = self.generate_batch_variations(current_batch, batch_size)
                        
                        for new_input, orig_item in zip(new_inputs, current_items):
                            if new_input:  # 빈 문자열이 아닌 경우만 처리
                                # 프롬프트 부분 제거
                                new_input = new_input.split("새로운 입력:")[-1].strip()
                                if new_input:  # 여전히 내용이 있는 경우
                                    augmented_data.append({
                                        'input': new_input,
                                        'output': orig_item['output']
                                    })
                            else:
                                failed_generations += 1
                                
                    except Exception as e:
                        logging.error(f"배치 처리 중 오류 발생: {str(e)}")
                        failed_generations += len(current_batch)
                    
                    current_batch = []
                    current_items = []
        
        # 남은 배치 처리
        if current_batch:
            try:
                new_inputs = self.generate_batch_variations(current_batch)
                for new_input, item in zip(new_inputs, current_items):
                    if new_input:
                        new_input = new_input.split("새로운 입력:")[-1].strip()
                        if new_input:
                            augmented_data.append({
                                'input': new_input,
                                'output': item['output']
                            })
                    else:
                        failed_generations += 1
            except Exception as e:
                logging.error(f"마지막 배치 처리 중 오류 발생: {str(e)}")
                failed_generations += len(current_batch)
        
        logging.info(f"생성 실패 횟수: {failed_generations}")
        return {'dataset': augmented_data}

if __name__ == "__main__":
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        sys.path.append(project_root)

        raw_data_path = os.path.join(project_root, 'ai', 'data', 'raw', 'simplified_generated_dataset.json')
        augmented_data_path = os.path.join(project_root, 'ai', 'data', 'augmented', 'openllama_dataset.json')

        with open(raw_data_path, 'r') as f:
            original_dataset = json.load(f)
        
        augmenter = OpenLLaMA_Augmenter()
        augmented_dataset = augmenter.augment_dataset(original_dataset, num_variations=2, batch_size=4)
        
        os.makedirs(os.path.dirname(augmented_data_path), exist_ok=True)
        with open(augmented_data_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_dataset, f, indent=4, ensure_ascii=False)
        
        logging.info(f"원본 데이터셋 크기: {len(original_dataset['dataset'])}")
        logging.info(f"증강된 데이터셋 크기: {len(augmented_dataset['dataset'])}")
        
    except Exception as e:
        logging.error(f"메인 실행 중 오류 발생: {str(e)}")
        sys.exit(1)