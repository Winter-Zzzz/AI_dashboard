from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import sys
import os
from pathlib import Path
from urllib.parse import unquote

# inference.py가 있는 디렉토리를 시스템 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import generate_code, transform_code, execute_code 
from config.model_config import ModelConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 관련 전역 변수
MODEL = None
TOKENIZER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer():
    """모델과 토크나이저 로드 함수"""
    config = ModelConfig()
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model')
    
    try:
        # 토크나이저 초기화
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            model_max_length=config.MAX_LENGTH,
            padding_side='right',
            truncation_side='right',
            legacy=False)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        # 특수 토큰 추가
        special_tokens = {
            'additional_special_tokens': [
                '<hex>', '</hex>', '<time>', '</time>'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # 모델 로드
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(DEVICE)
        
        print(f"모델 로딩 완료! (Device: {DEVICE})")
        return model, tokenizer

    except Exception as e:
        print(f"모델 로딩 중 에러 발생: {str(e)}")
        return None, None

@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 로드"""
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_model_and_tokenizer()

@app.get("/api/query-transactions")
async def query_transactions(query_text: str, dataset: str):
    """트랜잭션 쿼리 처리"""
    if not all([MODEL, TOKENIZER]):
        raise HTTPException(status_code=500, detail="모델이 초기화되지 않았습니다.")
    
    try:
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
        generated = generate_code(query_text, MODEL, TOKENIZER)
        transformed = transform_code(generated)
        result = execute_code(transformed, dataset_json)
        
        return {
            "status": "success",
            "data": {
                "transactions": result,
                "generated_code": transformed,
                "query_text": query_text  # 원본 쿼리도 포함
            }
          }   
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)