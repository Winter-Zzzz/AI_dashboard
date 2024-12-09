import logging
import nlpaug.augmenter.word as naw
from typing import List, Tuple, Set
import nltk
import re
import random
import time
import json
import os
import sys
from typing import List
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)

from utils.data_loader import load_training_data


class QueryAugmenterNlpAug:
    """입출력 텍스트를 토큰화하여 모델 학습용 데이터셋 생성"""
    def __init__(self):
        self._download_nltk_data()
        self._init_patterns()
        self._init_special_tokens()

        # 원본 텍스트에 대한 참조와 카운터들
        self.origin_text = []
        self.hex_counter = 1
        self.time_counter = 1
        self.func_counter = 1
        self.num_counter = 1
        
    def _download_nltk_data(self):
        """NLTK 데이터 패키지가 설치되어 있는지 확인하고 없으면 다운로드"""
        packages = ['wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt']
        for package in packages:
            try:
                nltk.data.find(f'{package}')
            except LookupError:
                nltk.download(package)

    def _init_patterns(self):
        self.patterns = {
        'HEX': re.compile(r'[0-9a-fA-F]{130}'),  # 130자리 16진수 값
        'TIME': re.compile(r'\b(?P<timestamp>\d{10})\b'),
        'FUNC': re.compile(r'\b\w+\s+function\b'),  # 함수 호출 패턴 추가
        'MOST': re.compile(r'most\srecent', re.IGNORECASE),  # most recent 처리
        'NUM': None  # 나중에 숫자 패턴을 설정할 것입니다.
    }

    def clean_output_text(self, text: str) -> str:
        """증강 데이터에 대한 기본적인 텍스트 정제"""
        # 메서드 체인 사이의 공백 제거
        text = re.sub(r'\s*\.\s*', '.', text)
        # 괄호 주변 공백 정리
        text = re.sub(r'\(\s*', '(', text)
        text = re.sub(r'\s*\)', ')', text)
        return text.strip()
    
    def _init_special_tokens(self):
        """토크나이저에서 보존해야 할 특수 토큰들 정의"""

        self.direction_tokens = ['to', 'from', 'by', 'all']
        self.temporal_tokens = ['latest', 'oldest', 'earliest', 'recent', 'after', 'before', 'between']

        self.number_variations = {
            1: ['one', '1'],
            2: ['two', '2'],
            3: ['three', '3'],
            4: ['four', '4'],
            5: ['five', '5'],
            6: ['six', '6'],
            7: ['seven', '7'],
            8: ['eight', '8'],
            9: ['nine', '9'],
            10: ['ten', '10']
        }

        # 특수 토큰들을 stopwords에 추가
        self.stopwords = set(self.direction_tokens + self.temporal_tokens)
        
        # 숫자 변형에 대한 패턴을 추가
        number_words = '|'.join(item for sublist in self.number_variations.values() for item in sublist)
        self.patterns['NUM'] = re.compile(
            rf'\b(?!(?:[0-9a-fA-F]{{130}}|\d{{10}})\b)({number_words}|\d+)\b',
            re.IGNORECASE
        )

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 - 태그를 임시 텍스트로 변환"""
        text = re.sub(r'\bmost recent\b', '__MOST_RECENT__', text)

        # HEX 값 변환 및 리스트에 저장
        text = re.sub(self.patterns['HEX'], 
                      lambda match: self._store_hex(match.group(0)), text)

        # TIME 값 변환 및 리스트에 저장
        text = re.sub(self.patterns['TIME'], 
                      lambda match: self._store_time(match.group(0)), text)

        # FUNC 값 변환 및 리스트에 저장
        text = re.sub(self.patterns['FUNC'], 
                      lambda match: self._store_func(match.group(0)), text)

        # NUM 값 변환 및 리스트에 저장
        text = re.sub(self.patterns['NUM'], 
                      lambda match: self._store_num(match.group(0)), text)

        return text
    
    def _store_hex(self, value: str) -> str:
        """HEX 값을 저장하고 변환된 텍스트 반환"""
        self.hex_values.append(value)  # HEX 값을 리스트에 저장
        self.hex_counter += 1  # 카운터 증가

        return f"HEX_{self.hex_counter}_TOKEN"
    
    def _store_time(self, value: str) -> str:
        """TIME 값을 저장하고 변환된 텍스트 반환"""
        self.time_values.append(value)  # TIME 값을 리스트에 저장
        self.time_counter += 1
        return f"TIME_{self.time_counter}_TOKEN"
    
    def _store_func(self, value: str) -> str:
        """FUNC 값을 저장하고 변환된 텍스트 반환"""
        self.func_values.append(value)  # FUNC 값을 리스트에 저장
        self.func_counter += 1
        return f"FUNC_{self.func_counter}_TOKEN"
    
    def _store_num(self, value: str) -> str:
        """NUM 값을 저장하고 변환된 텍스트 반환"""
        self.num_values.append(value)  # NUM 값을 리스트에 저장
        self.num_counter += 1
        return f"NUM_{self.num_counter}_TOKEN"

    def postprocess_text(self, text: str) -> str:
        """임시 텍스트를 원래 값으로 복원"""
        def restore_token(match):
            """임시 텍스트를 원래 값으로 복원"""
            token = match.group(1)
            token_type, index_str = token.split('_', 1)  # 첫 번째 '_' 기준으로 나눈 후 두 번째 부분 (value) 반환
            index = int(index_str) - 1  # 카운터는 1부터 시작했으므로 리스트 인덱스는 0부터 시작
            
            if token_type == "HEX":
                # HEX 값 복원
                return self.hex_values[index] if index < len(self.hex_values) else token
            elif token_type == "TIME":
                # TIME 값 복원
                return self.time_values[index] if index < len(self.time_values) else token
            elif token_type == "FUNC":
                # FUNC 값 복원
                return self.func_values[index] if index < len(self.func_values) else token
            elif token_type == "NUM":
                # NUM 값 복원
                return self.num_values[index] if index < len(self.num_values) else token
            return token  # 기본적으로 토큰을 그대로 반환

        # 'HEX_1234_TOKEN'과 같은 토큰을 원래 값으로 복원
        text = re.sub(r'([A-Z]+_\d+)_TOKEN', restore_token, text)
        # text = re.sub(r'(_token|_end)', '', text)  # '_token'과 '_end'를 제거
        text = re.sub(r'(\s+)', ' ', text)  # 여백 처리
        text = text.strip()  # 양옆 공백 제거
        text = text.lower()  # 소문자로 변환

        return text

    
    def initialize_augmenter(self, texts: List[str]):
        """입력된 텍스트 리스트를 사용하여 증강기를 초기화함"""
        combined_text = " ".join(texts)
        words = combined_text.split()
        
        # 정지어 생성
        stopwords = self.stopwords
        
        try:
            self.augmenters = [
                naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=0.1,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    device='cuda',
                    action="substitute",
                    aug_p=0.1,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.RandomWordAug(
                    action="swap",
                    aug_p=0.1,
                    aug_min=1,
                    stopwords=stopwords
                )
            ]
            logging.info("Augmenters initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize augmenters: {str(e)}")
            raise

    
    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2, batch_size: int=32) -> Tuple[List[str], List[str]]:
        augmented_inputs = []
        augmented_outputs = []

        # 전처리된 텍스트로 augmenter 초기화
        preprocessed_texts = [self.preprocess_text(text) for text in input_texts]
        self.initialize_augmenter(preprocessed_texts)

        for i in range(0, len(input_texts), batch_size):
            batch_inputs = input_texts[i:i + batch_size]  # 원본 텍스트 사용
            batch_preprocessed = preprocessed_texts[i:i + batch_size]  # 전처리된 텍스트
            batch_outputs = output_texts[i:i + batch_size]

            for idx, (orig_input, preprocessed_input, output_text) in enumerate(
                tqdm(zip(batch_inputs, batch_preprocessed, batch_outputs), 
                    total=len(batch_inputs),
                    desc=f"Batch {i//batch_size + 1}/{len(input_texts)//batch_size + 1}")
            ):
                try:
                    # 원본 입력도 보존
                    augmented_inputs.append(orig_input)
                    augmented_outputs.append(self.clean_output_text(output_text))
                    
                    for augmenter in self.augmenters:
                        try:
                            # 전처리된 텍스트로 증강
                            variations = augmenter.augment(preprocessed_input)
                            logging.info(f"Variations: {variations}")  # 반환값을 로그로 출력

                            if not variations:
                                logging.warning(f"No variations generated for input: {preprocessed_input}")
                                continue

                            for var in variations:
                                if 'UNK' not in var:
                                    try:
                                        # 각 변형에 대해 후처리 적용
                                        processed_var = self.postprocess_text(var)
                                        logging.info(f"Processed variation: {processed_var}")
                                        
                                        if processed_var != orig_input and processed_var not in augmented_inputs:
                                            augmented_inputs.append(processed_var)
                                            if output_text is not None:
                                                augmented_outputs.append(self.clean_output_text(output_text))
                                            else:
                                                logging.warning(f"Output text is None for input: {preprocessed_input}")
                                    except Exception as e:
                                        logging.error(f"Postprocessing failed for variation: {var} | Error: {str(e)}")
                                        continue
                                else:
                                    logging.info(f"Skipping variation with 'UNK': {var}")

                        except Exception as e:
                            logging.warning(f"Augmentation failed: {str(e)}")

                    logging.info(f"Processing batch {idx+1}/{len(batch_inputs)}")
                except Exception as e:
                    logging.error(f"Error at index {idx}: {str(e)}")
                    continue

        # 중복 제거
        unique_pairs = {}  # 입력에 대한 출력을 저장할 딕셔너리

        for inp, out in zip(augmented_inputs, augmented_outputs):
            # 각 입력에 대해 출력이 없거나, 이미 존재하는 입력이 아닌 경우에만 추가
            if inp not in unique_pairs:
                unique_pairs[inp] = out

        # 중복 제거된 결과 반환
        if unique_pairs:
            # 중복 제거된 입력과 출력을 반환
            augmented_inputs = list(unique_pairs.keys())
            augmented_outputs = list(unique_pairs.values())
            return augmented_inputs, augmented_outputs

        return [], []
        
def save_augmented_data(augmented_inputs: List[str], augmented_outputs: List[str], file_path: str):
    """증강된 데이터를 augmented_dataset.json으로 저장"""

    # 데이터셋 구조 생성
    dataset = [{'input': inp, 'output': out} for inp, out in zip(augmented_inputs, augmented_outputs)]
    data = {
        "dataset": dataset,
    }

    # JSON 파일로 저장
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Augmented data saved to {file_path}")

    # 통계 출력
    print("\n=== 증강 통계 ===")
    print(f"증강된 데이터 수: {len(augmented_inputs)}")
    print(f"저장 위치: {os.path.abspath(file_path)}")
    
    return data

if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)

        # 프로젝트 루트 경로 설정 및 sys.path에 추가
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        sys.path.append(project_root)

        # 데이터셋 경로 설정
        raw_data_path = os.path.join(project_root, 'ai', 'data', 'raw', 'simplified_generated_dataset.json')
        augmented_data_path = os.path.join(project_root, 'ai', 'data', 'augmented', 'simplified_augmented_dataset.json')

        # 원본 데이터 로드
        try:
            input_texts, output_texts = load_training_data(raw_data_path)

            # 데이터 검증
            if not isinstance(input_texts, list) or not isinstance(output_texts, list):
                raise ValueError("Training data must be lists.")

            if any(not isinstance(text, str) for text in input_texts):
                raise ValueError("All input data must be strings.")

            if any(not isinstance(text, str) for text in output_texts):
                raise ValueError("All output data must be strings.")
            
        except Exception as e:
            logging.error(f"Error loading or validating training data: {e}")
            sys.exit(1)  # 에러 발생 시 종료

        logging.info(f"Loaded {len(input_texts)} input-output pairs for augmentation.")

        augmenter = QueryAugmenterNlpAug()

        augmented_inputs, augmented_outputs = augmenter.augment(input_texts, output_texts, 1, 128)

        # 증강된 데이터를 저장
        try:
            save_augmented_data(augmented_inputs, augmented_outputs, augmented_data_path)
            logging.info(f"Augmented data saved to {augmented_data_path}")
        except Exception as e:
            logging.error(f"Error saving augmented data: {e}")
            sys.exit(1)
