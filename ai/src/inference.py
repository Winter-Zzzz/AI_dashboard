import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

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
        # 테스트용 더미 데이터 반환
        return {
            "transactions": [
                {"date": "2024-01-01", "amount": 100, "type": "income"},
                {"date": "2024-01-02", "amount": -50, "type": "expense"}
            ]
        }
    except Exception as e:
        print(f"JSON 파일 로드 중 에러 발생: {str(e)}")
        return None

def generate_code(input_text: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> str:
    """코드 생성 함수"""
    # 입력 준비
    input_text = f"Generate JavaScript: {input_text}"
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        padding=True, 
        truncation=True
    )
    
    # 생성
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
    
    # 디코딩
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_code.startswith("Generate JavaScript: "):
        generated_code = generated_code[len("Generate JavaScript: "):]
    
    return generated_code.strip()

def execute_js_code(code: str, data: dict):
    """JavaScript 코드 실행을 위한 Node.js 실행 함수"""
    try:
        import node_vm2
        vm = node_vm2.VM()
        
        full_code = f"""
            const data = {json.dumps(data, indent=2)};
            {code}
        """
        
        result = vm.run(full_code)
        print("\n실행 결과:")
        print(result)
    except ImportError:
        print("\nNode.js 실행을 위해서는 node-vm2 패키지가 필요합니다.")
        print("npm install -g node-vm2 로 설치할 수 있습니다.")
        print("\n대신 실행할 전체 코드를 출력합니다:")
        print("```javascript")
        print(f"const data = {json.dumps(data, indent=2)};\n")
        print(code)
        print("```")

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
            print("\n프로그램을 종료됩니다.")
            break
            
        if not user_input:
            print("질문을 입력해주세요!")
            continue
            
        try:
            print("\n생성 중...")
            generated_code = generate_code(user_input, model, tokenizer)
            print("\n생성된 코드:")
            print("```javascript")
            print(generated_code)
            print("```\n")

            execute_js_code(generated_code, data)
        except Exception as e:
            print(f"\n에러 발생: {str(e)}")

def main():
    """메인 함수"""
    config = ModelConfig()
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model')
    
    # 테스트 디렉토리 생성
    test_dir = create_test_directory()
    
    # JSON 데이터 로드
    json_path = os.path.join(os.path.dirname(__file__), 'test', 'transaction_test.json')
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"JSON 파일 경로: {json_path}")
    
    data = load_json_data(json_path)
    
    # 모델 로드
    try:
        print("학습된 모델을 로딩하는 중...")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")  # legacy 파라미터 제거
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        print("학습된 모델 로딩 완료!")
    except Exception as e:
        print(f"모델 로딩 중 에러 발생: {str(e)}")
        return

    model.eval()
    
    # 대화형 세션 시작
    interactive_session(model, tokenizer, data)

if __name__ == "__main__":
    main()