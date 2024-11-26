import json
from typing import Tuple, List
import logging
import os

def load_training_data(file_path: str) -> Tuple[List[str], List[str]]:
   """
   학습 데이터를 로드하는 함수
   
   Args:
       file_path: 데이터셋 JSON 파일 경로
       
   Returns:
       Tuple[List[str], List[str]]: (입력 텍스트 리스트, 출력 코드 리스트) 튜플
   """
   try:
       # 파일 경로 디버깅
       current_dir = os.getcwd()
       full_path = os.path.abspath(file_path)
       print(f"Current directory: {current_dir}")
       print(f"Attempting to load file: {full_path}")

       with open(file_path, 'r', encoding='utf-8') as f:
           data = json.load(f)
           
       # dataset 키로 접근
       dataset = data.get('dataset', [])
       
       input_texts = []
       output_texts = []
       
       # dataset 내의 각 항목에서 input과 output 추출
       for item in dataset:
           if 'input' in item and 'output' in item:
               input_text = item['input'].strip()
               output_text = item['output'].strip()
               
               # 빈 데이터 제외
               if input_text and output_text:
                   input_texts.append(input_text)
                   output_texts.append(output_text)
               else:
                   logging.warning("빈 input 또는 output 항목이 발견되었습니다")
           else:
               logging.warning("데이터 항목에 input 또는 output 키가 없습니다")
               
       if not input_texts or not output_texts:
           logging.error("유효한 데이터를 찾을 수 없습니다")
           return [], []
           
       if len(input_texts) != len(output_texts):
           logging.error(f"input과 output 개수가 일치하지 않습니다: inputs={len(input_texts)}, outputs={len(output_texts)}")
           return [], []
           
       # 데이터 검증
       for idx, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
           if not isinstance(input_text, str) or not isinstance(output_text, str):
               logging.error(f"잘못된 데이터 타입 발견 (index {idx})")
               return [], []
               
           if len(input_text.strip()) == 0 or len(output_text.strip()) == 0:
               logging.error(f"빈 데이터 발견 (index {idx})")
               return [], []
               
       logging.info(f"데이터 로드 완료: {len(input_texts)} samples")
       
       # 로드된 데이터 샘플 로깅
       if input_texts:
           logging.info(f"첫 번째 데이터 샘플:")
           logging.info(f"Input: {input_texts[0][:100]}...")
           logging.info(f"Output: {output_texts[0][:100]}...")
           
       return input_texts, output_texts
       
   except FileNotFoundError:
       logging.error(f"파일을 찾을 수 없습니다: {file_path}")
       return [], []
   except json.JSONDecodeError:
       logging.error(f"잘못된 JSON 형식: {file_path}")
       return [], []
   except Exception as e:
       logging.error(f"데이터 로드 중 에러 발생: {str(e)}")
       return [], []