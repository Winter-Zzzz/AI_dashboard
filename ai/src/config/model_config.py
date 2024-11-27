class ModelConfig:
    # 모델 선택
    MODEL_NAME = 't5-base'  # 더 큰 모델로 변경
    
    # 시퀀스 길이
    MAX_LENGTH = 512  # pk 2개(256자) + 명령어 텍스트 고려
    MAX_GEN_LENGTH = 512  # 생성되는 코드의 길이도 동일하게 설정
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 4 
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    NUM_EPOCHS = 5
    PATIENCE = 5
    
    # 생성 파라미터
    NUM_BEAMS = 10
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # 최적화 파라미터
    GRADIENT_CLIP = 0.5
    WARM_UP_RATIO = 0.1  # warmup 단계 줄임
    WARMUP_STEPS = None  # 제거
