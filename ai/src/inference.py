import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.filter_data import TransactionFilter

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

try:
    from config.model_config import ModelConfig
except ImportError:
    class ModelConfig:
        def __init__(self):
            self.MODEL_NAME = "t5-base"

def create_test_directory():
    """테스트 디렉토리 생성"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test')
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def load_json_data(json_path: str) -> dict:
    """JSON 데이터 로드 함수"""
    try:
        print(f"JSON 파일 경로: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON 파일을 찾을 수 없습니다: {json_path}")
        return {"transactions": []}
    except Exception as e:
        print(f"JSON 파일 로드 중 에러 발생: {str(e)}")
        return None

def generate_code(input_text: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> str:
    """코드 생성 함수"""
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        padding=True, 
        truncation=True
    )
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,
            num_beams=5,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    outputs = outputs.cpu()
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_code.strip()

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
    config = ModelConfig()
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model')
    
    test_dir = create_test_directory()
    
    json_path = os.path.join(os.path.dirname(__file__), 'test', 'transaction_test.json')
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"JSON 파일 경로: {json_path}")
    
    data = load_json_data(json_path)
    
    try:
        print("학습된 모델을 로딩하는 중...")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model = model.to(DEVICE)
        print(f"학습된 모델 로딩 완료! (Device: {DEVICE})")
    except Exception as e:
        print(f"모델 로딩 중 에러 발생: {str(e)}")
        return

    model.eval()
    
    interactive_session(model, tokenizer, data)

if __name__ == "__main__":
    main()