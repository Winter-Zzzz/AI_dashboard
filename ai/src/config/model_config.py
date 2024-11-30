class ModelConfig:
    # 모델 선택
    MODEL_NAME = 't5-small'
    
    # 시퀀스 길이
    MAX_LENGTH = 256
    MAX_GEN_LENGTH = 256
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 15
    PATIENCE = 7
    
    # 생성 파라미터
    NUM_BEAMS = 5
    LENGTH_PENALTY=0.6
    NO_REPEAT_NGRAM_SIZE=2
    
    # 최적화 파라미터
    GRADIENT_CLIP = 1.0

    ACCUMULATION_STEPS = 4

    EARLY_STOPPING = True