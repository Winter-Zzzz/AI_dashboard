class ModelConfig:
    # 기본 모델 설정
    MODEL_NAME = 't5-base'  # small -> base로 변경 
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 20
    PATIENCE = 3
    
    # 생성 파라미터
    MAX_GEN_LENGTH = 1024
    NUM_BEAMS = 5
    TEMPERATURE = 0.7
    TOP_P = 0.95

    # 추가된 최적화 파라미터
    GRADIENT_CLIP = 1.0
    WARM_UP_RATIO = 0.1
    WARMUP_STEPS = 0