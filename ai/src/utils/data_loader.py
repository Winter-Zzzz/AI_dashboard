import json
from typing import Tuple, List
import logging
import os
from .preprocessing import preprocess_input_text, preprocess_output_text

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
        logging.info(f"Attempting to load file: {os.path.abspath(file_path)}")

        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 'dataset' 키로 접근
        dataset = data.get('dataset', [])
        if not isinstance(dataset, list):
            logging.error("dataset 키가 리스트가 아닙니다.")
            return [], []

        # input/output 추출
        input_texts = []
        output_texts = []

        for idx, item in enumerate(dataset):
            try:
                # input/output 검증
                input_text = item.get('input', '').strip()
                output_text = item.get('output', '').strip()

                if not input_text or not output_text:
                    logging.warning(f"빈 input/output 항목 발견 (index {idx}): {item}")
                    continue

                processed_input = preprocess_input_text(input_text)
                processed_output = preprocess_output_text(output_text)

                input_texts.append(processed_input)
                output_texts.append(processed_output)

            except Exception as e:
                logging.error(f"데이터 처리 중 에러 발생 (index {idx}): {str(e)}")
                continue

        # 검증 및 로깅
        if not input_texts or not output_texts:
            logging.error("유효한 데이터를 찾을 수 없습니다.")
            return [], []

        if len(input_texts) != len(output_texts):
            logging.error(f"input과 output의 길이가 일치하지 않습니다. inputs={len(input_texts)}, outputs={len(output_texts)}")
            return [], []

        for idx, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            if not isinstance(input_text, str) or not isinstance(output_text, str):
                logging.error(f"잘못된 데이터 타입 발견 (index {idx}). input={input_text}, output={output_text}")
                return [], []

        logging.info(f"데이터 로드 완료: {len(input_texts)} samples")

        # 로드된 데이터 샘플 로깅
        logging.info(f"첫 번째 데이터 샘플:")
        if input_texts:
            logging.info(f"Input: {input_texts[0][:100]}...")
            logging.info(f"Output: {output_texts[0][:100]}...")

        return input_texts, output_texts

    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {file_path}")
        return [], []
    except json.JSONDecodeError as e:
        logging.error(f"잘못된 JSON 형식: {file_path}. Error: {str(e)}")
        return [], []
    except Exception as e:
        logging.error(f"데이터 로드 중 에러 발생: {str(e)}")
        return [], []
