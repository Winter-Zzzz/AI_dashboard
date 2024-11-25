import json
from typing import Tuple, List
import logging

def load_training_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    학습 데이터를 로드하는 함수
    
    Args:
        file_path: 데이터셋 JSON 파일 경로
        
    Returns:
        Tuple[List[str], List[str]]: (입력 텍스트 리스트, 출력 텍스트 리스트) 튜플
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'dataset' not in data:
            logging.error(f"잘못된 JSON 형식 (dataset 키 없음): {file_path}")
            return [], []
            
        dataset = data['dataset']
        if not dataset:
            logging.error(f"빈 데이터셋: {file_path}")
            return [], []
            
        input_texts = []
        output_texts = []
        
        for item in dataset:
            if 'input' in item and 'output' in item:
                input_texts.append(item['input'])
                output_texts.append(item['output'])
            else:
                logging.warning("데이터 항목에 input 또는 output 키가 없습니다")
                
        if not input_texts or not output_texts:
            logging.error("유효한 데이터를 찾을 수 없습니다")
            return [], []
            
        if len(input_texts) != len(output_texts):
            logging.error(f"입력과 출력 개수가 일치하지 않습니다: inputs={len(input_texts)}, outputs={len(output_texts)}")
            return [], []
            
        logging.info(f"데이터 로드 완료: {len(input_texts)} samples")
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