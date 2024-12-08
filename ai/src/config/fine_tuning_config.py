import os
class ModelConfig:

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 모델 설정
    BASE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model")
    MODEL_NAME = "t5-small"  # 폴백용 기본 모델
    USE_BASE_MODEL = True  # 기존 모델 사용 여부
    
    # Fine-tuning 파라미터
    MAX_LENGTH = 256
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # 데이터 처리
    MAX_INPUT_LENGTH = 256
    MAX_TARGET_LENGTH = 256
    
    # 학습 설정
    LOGGING_STEPS = 10
    EVAL_STEPS = 100
    SAVE_STRATEGY = "epoch"
    EVAL_STRATEGY = "epoch"
    
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
    
    # 생성 설정
    GENERATION_CONFIG = {
        'num_beams': 4,
        'early_stopping': True,
        'max_length': 128
    }
