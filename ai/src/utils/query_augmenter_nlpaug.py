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
import json
from typing import List
from tqdm import tqdm

# data_loader.py에서 load_training_data 함수 가져오기 
from data_loader import load_training_data

class QueryAugmenterNlpAug:
    """입출력 텍스트를 토큰화하여 모델 학습용 데이터셋 생성"""
    def __init__(self):
        self._download_nltk_data()
        self._init_patterns()
        self._init_preserved_keywords()
        self._init_number_variations()

    def _download_nltk_data(self):
        """NLTK 데이터 패키지가 설치되어 있는지 확인하고 없으면 다운로드"""
        packages = ['wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt']
        for package in packages:
            try:
                nltk.data.find(f'{package}')
            except LookupError:
                nltk.download(package)

    def _init_patterns(self):
        """텍스트에서 특정 패턴을 식별하기 위한 정규 표현식 초기화"""
        self.patterns = {
            'hex': re.compile(r'[0-9a-fA-F]{130}'), # 130자리 16진수 값
            'from': re.compile(r'from\s+([0-9a-fA-F]{130})'), # from [hex]
            'to': re.compile(r'to\s+([0-9a-fA-F]{130})'), # to [hex]
            'timestamp': re.compile(r'\b\d{10}\b'), 
            'func_name': re.compile(r'\b(setup function|on function|off function)\b', re.IGNORECASE) 
        }
    
    def _init_preserved_keywords(self):
        """모델 처리 중 변경하지 말아야 할 키워드"""
        self.preserved_keywords = {
            'setup function', 'on function', 'off function', 'to', 'from'
        }

    def _init_number_variations(self):
        """숫자를 다양한 텍스트 표현과 매칭되도록 처리함"""
        self.number_variations = { # 숫자 변형 정의
            1: ['one', 'single', '1'],
            2: ['two', 'couple', '2'],
            3: ['three', '3'],
            4: ['four', '4'],
            5: ['five', '5'],
            10: ['ten', '10']
        }
        number_words = '|'.join(item for sublist in self.number_variations.values() for item in sublist)
        self.patterns['number'] = re.compile(
            rf'\b(?!(?:[0-9a-fA-F]{{130}}|\d{{10}})\b)({number_words}|\d+)\b',
            re.IGNORECASE
        )

    def is_preserved(self, word: str) -> bool:
        """주어진 키워드가 보존되어야 하는 키워드인지 확인"""
        return (
            word in self.preserved_keywords or
            any(pattern.match(word) for pattern in [self.patterns[key] for key in ['hex', 'from', 'to', 'timestamp']])
    ) 

    def get_stopwords_from_preserved(self, text: list) -> list:
        """정지어 리스트 추출"""
        stopwords = [word for word in text if self.is_preserved(word)]
        return stopwords

    def initialize_augmenter(self, texts: List[str]):
        """입력된 텍스트 리스트를 사용하여 증강기를 초기화함"""

        # 텍스트 결합 - 입력된 텍스트를 하나로 결합하고 이를 단어 단위로 나눔
        combined_text = " ".join(texts)
        words = combined_text.split()

        # 정지어 생성 
        stopwords = self.get_stopwords_from_preserved(words)

        try:
            self.augmenters = [
                naw.SynonymAug( # WordNet을 사용해 동의어 치환 수행
                    aug_src='wordnet',
                    aug_p=0.2,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.ContextualWordEmbsAug( # BERT 모델을 사용해 문맥에 따라 단어 대체
                    model_path='bert-base-uncased',
                    device='cuda',
                    action="substitute",
                    aug_p=0.2,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.RandomWordAug( # BERT 모델을 사용해 문맥에 따라 
                    action="swap",
                    aug_p=0.3,
                    aug_min=1,
                    stopwords=stopwords
                )
            ]
            logging.info("Augmenters initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize augmenters: {str(e)}")
            raise

    def validate_function_keywords(self, text: str) -> bool:
        """입력 텍스트의 function 키워드 유효성 검사"""
        # off나 setup이 중복으로 나오는지만 체크
        words = text.lower().split()
        off_count = words.count('off')
        setup_count = words.count('setup')
        
        # off나 setup이 1번 초과로 나오면 거부
        return off_count <= 1 and setup_count <= 1

    def clean_output_text(self, text: str) -> str:
        """증강 데이터에 대한 기본적인 텍스트 정제"""
        # 메서드 체인 사이의 공백 제거
        text = re.sub(r'\s*\.\s*', '.', text)
        # 괄호 주변 공백 정리
        text = re.sub(r'\(\s*', '(', text)
        text = re.sub(r'\s*\)', ')', text)
        return text.strip()
        
    def filter_augmented_pairs(self, inputs: List[str], outputs: List[str]) -> Tuple[List[str], List[str]]:
        """증강된 데이터 쌍 필터링"""
        filtered_inputs = []
        filtered_outputs = []
        
        for input_text, output_text in zip(inputs, outputs):
            # function 키워드 유효성 검사
            if self.validate_function_keywords(input_text):
                cleaned_output = self.clean_output_text(output_text)
                filtered_inputs.append(input_text)
                filtered_outputs.append(cleaned_output)
                
        return filtered_inputs, filtered_outputs
    
    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2, batch_size: int=32) -> Tuple[List[str], List[str]]:
        """입력 텍스트 변형과 해당 변형에 맞는 출력 텍스트 생성"""
        augmented_inputs = []
        augmented_outputs = []

        # 초기 데이터 정리 및 필터링
        input_texts, output_texts = self.filter_augmented_pairs(input_texts, output_texts)
        self.initialize_augmenter(input_texts)

        for i in range(0, len(input_texts), batch_size):
            batch_inputs = input_texts[i:i + batch_size]
            batch_outputs = output_texts[i:i + batch_size]

            for idx, (input_text, output_text) in enumerate(tqdm(zip(batch_inputs, batch_outputs), 
                                                                 total=len(batch_inputs),
                                                                 desc=f"Batch {i//batch_size + 1}/{len(input_texts)//batch_size + 1}")):
                try:
                    for augmenter in self.augmenters:
                        try:
                            variations = augmenter.augment(input_text, n=num_variations)

                            for var in variations:
                                if 'UNK' not in var and self.validate_function_keywords(var):
                                    augmented_inputs.append(var)
                                    augmented_outputs.append(output_text)
                        except Exception as e:
                            logging.warning(f"Augmentation failed: {str(e)}")
                            continue
                    logging.info(f"Processing batch {idx+1}/{len(input_texts)}")
                except Exception as e:
                    logging.error(f"Error at index {idx}: {str(e)}")
                    continue

        unique_pairs = list(dict.fromkeys(zip(augmented_inputs, augmented_outputs)))
        if unique_pairs:
            augmented_inputs, augmented_outputs = zip(*unique_pairs)
            return list(augmented_inputs), list(augmented_outputs)
        return [],[]
    
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

    augmented_inputs, augmented_outputs = augmenter.augment(input_texts, output_texts, 1, 512)

    # 증강된 데이터를 저장
    try:
        save_augmented_data(augmented_inputs, augmented_outputs, augmented_data_path)
        logging.info(f"Augmented data saved to {augmented_data_path}")
    except Exception as e:
        logging.error(f"Error saving augmented data: {e}")
        sys.exit(1)