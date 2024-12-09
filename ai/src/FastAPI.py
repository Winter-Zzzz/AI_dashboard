from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import sys
import os
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import generate_code, clean_generated_code, execute_code 
from config.model_config import ModelConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json

class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model_and_tokenizer(self):
        """모델과 토크나이저를 가져오는 메서드"""
        if self._model is None or self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._model, self._tokenizer
    
    def _load_model_and_tokenizer(self):
        """모델과 토크나이저를 로드하는 내부 메서드"""
        config = ModelConfig()
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model')
        checkpoint_path = os.path.join(model_path, 'model_checkpoint.pt')

        if not os.path.exists(model_path):
            raise Exception(f"모델 디렉토리가 존재하지 않습니다: {model_path}")
            
        if not os.path.exists(checkpoint_path):
            raise Exception(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
        
        try:

            print("학습된 모델을 로딩하는 중")

            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=True)

            self._tokenizer = T5Tokenizer.from_pretrained(model_path)
            
            self._model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
            self._model.resize_token_embeddings(len(self._tokenizer))
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model = self._model.to(self._device)
            
            print(f"학습된 모델 로딩 완료! (Device: {self._device})")
            
        except Exception as e:
            print(f"모델 로딩 중 에러 발생: {str(e)}")
            self._model = None
            self._tokenizer = None
            raise e

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/query-transactions")
async def query_transactions(query_text: str, dataset: str):
    """트랜잭션 쿼리 처리"""
    try:
        # ModelManager 인스턴스 가져오기
        model_manager = ModelManager()
        model, tokenizer = model_manager.get_model_and_tokenizer()
        
        if not all([model, tokenizer]):
            raise HTTPException(status_code=500, detail="모델 초기화에 실패했습니다.")
        
        if not query_text:
            raise HTTPException(status_code=400, detail="쿼리 텍스트가 필요합니다.")
            
        if not dataset:
            raise HTTPException(status_code=400, detail="데이터셋이 필요합니다.")
        
        try:
            decoded_dataset = unquote(dataset)
            dataset_json = json.loads(decoded_dataset)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="잘못된 데이터셋 형식입니다.")
        
        # inference.py의 함수들을 순차적으로 호출
        generated = generate_code(query_text, model, tokenizer)
        transformed = transform_code(generated)
        transformed = transformed.replace(" ", "")
        result = execute_code(transformed, dataset_json)
        
        if result is None:
            raise HTTPException(status_code=400, detail="쿼리 처리에 실패했습니다.")
        return {
            "status": "success",
            "data": {
                "transactions": result,
                "generated_code": transformed,
                "query_text": query_text
            }
        }   
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)