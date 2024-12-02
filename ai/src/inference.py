import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.simplified_filter_data import TransactionFilter

from pathlib import Path
from config.model_config import ModelConfig

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def load_json_data(file_path):
    """JSON 파일에서 데이터 로드"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"데이터 로드 완료: {file_path}")
        return data
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {str(e)}")
        return None

def generate_code(input_text: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> str:
    config = ModelConfig()
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True
    )
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=config.MAX_GEN_LENGTH,
            num_beams=config.NUM_BEAMS,
            length_penalty=config.LENGTH_PENALTY,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            early_stopping=config.EARLY_STOPPING
        )
    
    outputs = outputs.cpu()

    generated_code = tokenizer.decode(outputs[0])
    return generated_code


def execute_code(code: str, data: dict):
    """Python 코드 실행 함수"""
    try:
        print("\n실행 결과:")
        exec_globals = {
            "data": data,
            "TransactionFilter": TransactionFilter,  # TransactionFilter 클래스를 실행 환경에 추가
            "result": None
        }
        exec(code, exec_globals)
        return exec_globals.get("result")
    except Exception as e:
        print(f"코드 실행 중 에러 발생: {str(e)}")
        return None

def transform_code(code: str):
    """<pad> </s> 제거"""
    code = code.replace("<pad>", "")
    code = code.replace("</s>", "")
    # Remove extra whitespace
    code = " ".join(code.split())

    # dataset_generator  부터 실수함  고쳐야함 !!!
    if code.endswith("))"):
        code = code[:-1]

    if code.startswith("txn."):
        code = f"txn = TransactionFilter(data).reset()\nresult = {code}\nprint(result)"
    return code

def interactive_session(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, data: dict = None):
    """대화형 세션 실행"""
    if data is None:
        data = {"transactions": []}
        
    print("\n=== 코드 생성 AI 시스템 ===")
    print("'quit' 또는 'exit'를 입력하면 종료됩니다.")
    print("질문을 입력해주세요\n")
    
    while True:
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n프로그램을 종료합니다.")
            break
            
        if not user_input:
            print("질문을 입력해주세요!")
            continue
            
        try:
            print("\n생성 중...")
            generated_code = transform_code(generate_code(user_input, model, tokenizer))
            print("\n생성된 코드:")
            print("```python")
            print(generated_code)
            print("```\n")

            execute_code(generated_code, data)
        except Exception as e:
            print(f"\n에러 발생: {str(e)}")

def main():
    """메인 함수"""
    config = ModelConfig()  # CodeGeneratorConfig 대신 ModelConfig 사용
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model')
    
    # 테스트 데이터 로드
    json_path = os.path.join(PROJECT_ROOT, 'src', 'test', 'transaction_test.json')
    data = load_json_data(json_path)
    
    try:
        print("학습된 모델을 로딩하는 중...")
        # 토크나이저 먼저 초기화
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            model_max_length=config.MAX_LENGTH,
            padding_side='right',
            truncatino_side='right',
            legacy=False)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        # 특수 토큰 추가 - train.py와 동일한 토큰들 사용
        special_tokens = {
            'additional_special_tokens': [
                '<hex>', '</hex>', '<time>', '</time>'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # 모델 로드
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # 임베딩 레이어 리사이징 추가
        model.resize_token_embeddings(len(tokenizer))
        
        model = model.to(DEVICE)  # DEVICE 대신 device 사용 (상단에서 정의된 것)

        print(f"학습된 모델 로딩 완료! (Device: {DEVICE})")

        interactive_session(model, tokenizer, data)

    except Exception as e:
        print(f"모델 로딩 중 에러 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()