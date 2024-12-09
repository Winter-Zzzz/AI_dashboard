import os

class ModelConfig:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 모델 설정
    BASE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model")
    MODEL_NAME = "t5-small"  # 폴백용 기본 모델
    USE_BASE_MODEL = True  # 기존 모델 사용 여부
    
    # Fine-tuning 파라미터 - 최적화된 설정
    MAX_LENGTH = 256  # 길이 감소로 메모리 효율성 증가
    NUM_EPOCHS = 5
    BATCH_SIZE = 16  # 배치 사이즈 증가
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100  # 워밍업 스텝 증가
    ACCUMULATION_STEPS = 4  # 누적 스텝 감소
    
    # 데이터 처리
    MAX_INPUT_LENGTH = MAX_LENGTH
    MAX_TARGET_LENGTH = MAX_LENGTH
    
    # 학습 설정
    LOGGING_STEPS = 100
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "steps"
    GRADIENT_CLIP = 1.0
    
    # 성능 최적화 설정
    FP16 = True  # mixed precision 학습 활성화
    NUM_WORKERS = 2  # 데이터 로딩 병렬화
    PIN_MEMORY = True  # CUDA 핀 메모리 사용
    
    # 경로 설정
    OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "fine_tuned_model")
    LOGGING_DIR = os.path.join(ROOT_DIR, "logs")
    DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "simplified_generated_dataset.json")
    
    # 패턴 정의
    PATTERNS = {
        'hex': r'[0-9a-fA-F]{130}',
        'time': r'\b\d{10}\b',
        'func': r'\b\w+(?=\s+function\b)'
    }
    
    # 생성 설정 - 최적화된 설정
    GENERATION_CONFIG = {
        'num_beams': 5,
        'early_stopping': True,
        'max_length': 256,  # 생성 길이도 감소
        'no_repeat_ngram_size': 2,
        'length_penalty': 1.0
    }