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

# data_loader.py에서 load_training_data 함수 가져오기 
from data_loader import load_training_data

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

class QueryAugmenterNlpAug:
    def __init__(self):
        self._download_nltk_data()
        self._init_patterns()
        self._init_preserved_keywords()
        self._init_number_variations()
    
    def _download_nltk_data(self):
        packages = ['wordnet', 'averaged_perceptron_tagger', 'punkt']
        for package in packages:
            try:
                nltk.data.find(f'{package}')
            except LookupError:
                nltk.download(package)

    def _init_patterns(self):
        self.patterns = {
            'hex': re.compile(r'[0-9a-fA-F]{130}'),
            'from': re.compile(r'from\s+([0-9a-fA-F]{130})'),
            'to': re.compile(r'to\s+([0-9a-fA-F]{130})'),
            'timestamp': re.compile(r'\b\d{10}\b'),
            'func_name': re.compile(r'\b(setup function|on function|off function)\b', re.IGNORECASE)
        }
    
    def _init_preserved_keywords(self):
        self.preserved_keywords = {
            'setup funciton', 'on funciton', 'off function'
        }

    def _init_number_variations(self):
        self.number_variations = {
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

    def initialize_augmenter(self, texts: List[str]):
        combined_text = " ".join(texts)
        words = combined_text.split()
        stopwords = self.get_stopwords_from_preserved(words)

        try:
            self.augmenters = [
                naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=0.3,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    action="substitute",
                    aug_p=0.3,
                    aug_min=1,
                    stopwords=stopwords
                ),
                naw.RandomWordAug(
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

    def is_preserved(self, word: str) -> bool:
        """주어진 키워드가 보존되어야 하는 키워드인지 확인"""
        return (
            word in self.preserved_keywords or
            any(pattern.match(word) for pattern in [self.patterns[key] for key in ['hex', 'from', 'to', 'timestamp']])
    ) 


    def get_stopwords_from_preserved(self, text: list) -> list:
        stopwords = [word for word in text if self.is_preserved(word)]
        return stopwords

    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2) -> Tuple[List[str], List[str]]:
        augmented_inputs = []
        augmented_outputs = []
        self.initialize_augmenter(input_texts)

        for idx, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            try:
                for augmenter in self.augmenters:
                    try:
                        variations = augmenter.augment(input_text, n=num_variations)
                        # [UNK]
                        filtered_variations = [var for var in variations if 'UNK' not in var]
                        for var in filtered_variations:
                            augmented_inputs.append(var)
                            augmented_outputs.append(output_text)
                    except Exception as e:
                        logging.warning(f"Augmentation failed: {str(e)}")
                        continue
            except Exception as e:
                logging.error(f"Error at index {idx}: {str(e)}")
                continue

        unique_pairs = list(dict.fromkeys(zip(augmented_inputs, augmented_outputs)))
        if unique_pairs:
            augmented_inputs, augmented_outputs = zip(*unique_pairs)
            return list(augmented_inputs), list(augmented_outputs)
        return [],[]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 프로젝트 루트 경로 설정 및 sys.path에 추가
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(project_root)

    # 데이터셋 경로 설정
    raw_data_path = os.path.join(project_root, 'ai', 'data', 'raw', 'generated_dataset.json')
    augmented_data_path = os.path.join(project_root, 'ai', 'data', 'augmented', 'augmented_dataset.json')

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

    augmented_inputs, augmented_outputs = augmenter.augment(input_texts, output_texts)

    # 증강된 데이터를 저장
    try:
        save_augmented_data(augmented_inputs, augmented_outputs, augmented_data_path)
        logging.info(f"Augmented data saved to {augmented_data_path}")
    except Exception as e:
        logging.error(f"Error saving augmented data: {e}")
        sys.exit(1)