import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.filter_data import TransactionFilter

from pathlib import Path
from config.model_config import ModelConfig

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def clean_generated_text(text: str) -> str:
    # 특수 토큰 제거
    special_tokens = ["<pad>", "</s>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # 불필요한 공백 제거
    text = " ".join(text.split())
    
    # 괄호, 점 주변의 공백 제거
    text = text.replace(" (", "(")
    text = text.replace(" )", ")")
    text = text.replace(" .", ".")
    text = text.replace(" [", "[")
    text = text.replace(" ]", "]")
    text = text.replace(" :", ":")
    text = text.replace(" ,", ",")
    
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def generate_code(input_text: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> str:
    config = ModelConfig()
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=config.MAX_LENGTH,  # MAX_INPUT_LENGTH 대신 MAX_LENGTH 사용
        padding=True, 
        truncation=True
    )
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=config.MAX_GEN_LENGTH,
            num_beams=config.NUM_BEAMS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    outputs = outputs.cpu()

    generated_code = tokenizer.decode(outputs[0])
    return clean_generated_text(generated_code)


def execute_code(code: str, data: dict):
    """Python 코드 실행 함수"""
    try:
        print("\n실행 결과:")
        exec_globals = {
            "data": data,
            "TransactionFilter": TransactionFilter  # TransactionFilter 클래스를 실행 환경에 추가
        }
        exec(code, exec_globals)
    except Exception as e:
        print(f"코드 실행 중 에러 발생: {str(e)}")

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
            generated_code = generate_code(user_input, model, tokenizer)
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
    

    # test_dir = create_test_directory()
    # json_path = os.path.join(os.path.dirname(__file__), 'test', 'transaction_test.json')
    # data = load_json_data(json_path)
    
    try:
        print("학습된 모델을 로딩하는 중...")
        # 토크나이저 먼저 초기화
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        tokenizer.add_prefix_space = True
        
        # 특수 토큰 추가 - train.py와 동일한 토큰들 사용
        special_tokens = {
            'additional_special_tokens': [
                # 파이썬 기본 문법
                'def', 'class', 'return', 'import', 'from', 'print',
                # TransactionFilter 관련
                'TransactionFilter', 'by_pk', 'by_src_pk', 'by_func_name',
                'by_timestamp', 'sort', 'get_result',
                # 구분자
                '(', ')', '[', ']', '{', '}', '.', ',', ':', '"', "'",
                # 연산자
                '=', '==', '>', '<', '>=', '<=',
                # 자주 사용되는 키워드
                'data', 'reverse', 'True', 'False', 'None'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # 모델 로드
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # 임베딩 레이어 리사이징 추가
        model.resize_token_embeddings(len(tokenizer))
        
        model = model.to(DEVICE)  # DEVICE 대신 device 사용 (상단에서 정의된 것)

        print(f"학습된 모델 로딩 완료! (Device: {DEVICE})")
    except Exception as e:
        print(f"모델 로딩 중 에러 발생: {str(e)}")
        return

    model.eval()

    data = 0

    interactive_session(model, tokenizer, data)

if __name__ == "__main__":
    main()