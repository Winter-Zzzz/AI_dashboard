from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    # 모델 설정
    MODEL_NAME: str = 't5-small'  # 사용할 모델 이름 (Huggingface에서 제공하는 모델)
    MAX_LENGTH: int = 256         # 입력 텍스트의 최대 길이
    MAX_GEN_LENGTH: int = 256     # 생성된 텍스트의 최대 길이
    VOCAB_SIZE: int = 32100       # T5 tokenizer의 기본 vocab size
    
    # 학습 하이퍼파라미터
    BATCH_SIZE: int = 8           # 배치 크기
    LEARNING_RATE: float = 1e-5   # 학습률
    WEIGHT_DECAY: float = 0.01    # 가중치 감쇠 (L2 정규화)
    NUM_EPOCHS: int = 5           # 에폭 수
    PATIENCE: int = 3             # Early Stopping을 위한 patience
    
    # 생성 파라미터
    NUM_BEAMS: int = 5            # 빔 탐색 시 빔의 개수
    LENGTH_PENALTY: float = 1.0    # 길이 패널티
    NO_REPEAT_NGRAM_SIZE: int = 2  # 중복 n-gram 방지를 위한 크기
    
    # 최적화 파라미터
    GRADIENT_CLIP: float = 1.0     # 기울기 클리핑
    ACCUMULATION_STEPS: int = 4    # 기울기 누적 스텝
    WARMUP_RATIO: float = 0.1      # learning rate 워밍업 비율
    
    # 학습 제어
    EARLY_STOPPING: bool = True    # Early stopping 사용 여부
    SEED: int = 42                # 랜덤 시드
    FP16: bool = True             # 16비트 부동소수점 사용 여부
    
    def __post_init__(self):
        """디렉토리 경로 설정"""
        # 프로젝트 루트 디렉토리 설정
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 모델 관련 디렉토리
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, 'models')
        self.BEST_MODEL_DIR = os.path.join(self.MODEL_DIR, 'best_model')
        self.FINETUNED_MODEL_DIR = os.path.join(self.MODEL_DIR, 'fine_tuned_model')
        
        # 데이터 관련 디렉토리
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'data')
        self.AUGMENTED_DATA_DIR = os.path.join(self.DATA_DIR, 'augmented')
        
        # 로그 디렉토리
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        
        # 필요한 디렉토리 생성
        for directory in [
            self.MODEL_DIR, 
            self.BEST_MODEL_DIR,
            self.FINETUNED_MODEL_DIR,
            self.DATA_DIR,
            self.AUGMENTED_DATA_DIR,
            self.LOG_DIR
        ]:
            os.makedirs(directory, exist_ok=True)

    @property
    def device_settings(self):
        """GPU 관련 설정 반환"""
        import torch
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_gpu': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'fp16': self.FP16 and torch.cuda.is_available()
        }

    def to_dict(self):
        """설정값들을 dictionary 형태로 반환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}