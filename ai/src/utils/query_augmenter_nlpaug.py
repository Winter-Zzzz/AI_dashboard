import sys
import logging
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from typing import List, Tuple, Dict, Optional
import nltk
import re
import json
import os
from tqdm import tqdm
from torch.cuda import is_available as cuda_available
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QueryAugmenterNlpAug:
    PATTERNS = {
        'HEX': re.compile(r'[a-fA-F0-9]{130}'),
        'TS': re.compile(r'\d{10}'),
        'FUNC': re.compile(r'[a-zA-Z0-9_]+function', re.IGNORECASE)
    }
    ADDITIONAL_PATTERNS = {
        'HEXD': re.compile(r'\b(to|from|by)\s+[a-fA-F0-9]{130}\b'), 
        'TSD': re.compile(r'\b(after|before)\s+\d{10}\b') 

    }
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if cuda_available() else 'cpu')
        self.augmenters = None
        
    def _find_patterns(self, texts: List[str]) -> List[str]:
        """텍스트들에서 보존할 패턴들을 찾아서 반환"""
        preserved_words = set()
        additional_preserved_words = set()
        
        for text in texts:
            for pattern_type, regex in self.PATTERNS.items():
                matches = regex.finditer(text)
                for match in matches:
                    preserved_words.add(match.group())
                    additional_preserved_words.add(match.group())

        for text in texts:
            for pattern_type, regex in self.ADDITIONAL_PATTERNS.items():
                matches = regex.finditer(text)
                for match in matches:
                    additional_preserved_words.add(match.group())
                
        return list(preserved_words), list(additional_preserved_words)
        
    def _initialize_augmenters(self, texts: List[str]) -> None:
        """주어진 텍스트들에서 패턴을 찾아 stopwords로 설정"""
        try:
            stopwords, additional_stopwords = self._find_patterns(texts)
            logging.info(f"Found {len(stopwords)} patterns to preserve")
            
            # # 1. WordNet 기반 동의어 교체 -> BERT 기반이랑 겹쳐서 제거
            # synonym_aug = naw.SynonymAug(
            #     aug_src='wordnet',
            #     aug_p=0.3,
            #     stopwords=stopwords
            # )
            
            # 2. BERT 기반 문맥 교체
            context_aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                device=self.device,
                action="substitute",
                aug_p=0.3,
                stopwords=stopwords
            )

            # 3. Random Word Aug (단어 순서 섞기)
            word_swap_aug = naw.RandomWordAug(
                action='swap',  # 'swap'은 단어 순서를 바꿔줍니다.
                aug_p=0.3,  # 증강 확률
                stopwords=additional_stopwords
            )

            self.augmenters = [context_aug, word_swap_aug]

            
            # BackTranslation은 필요한 경우에만 별도로 처리
            # if cuda_available():
            #     back_translation_aug = naw.BackTranslationAug(
            #         from_model_name='facebook/wmt19-en-de',
            #         to_model_name='facebook/wmt19-de-en',
            #         device=self.device,
            #         max_length=512
            #     )
            #     self.augmenters.append(back_translation_aug)
            
            if not self.augmenters:  # 만약 증강기가 초기화되지 않았다면
                raise ValueError("No augmenters were initialized properly.")
            
        except Exception as e:
            logging.error(f"Augmenter initialization failed: {str(e)}")
            raise

    def augment(self, 
                input_texts: List[str], 
                output_texts: List[str], 
                num_augments_per_method: int = 1) -> Tuple[List[str], List[str]]:
        
            if len(input_texts) != len(output_texts):
                raise ValueError("Input and output text lists must have equal length")
                
            # 먼저 augmenters 초기화
            if self.augmenters is None:
                self._initialize_augmenters(input_texts)
                
            augmented_inputs = []
            augmented_outputs = []
            augmented_count = 0
            
            for input_text, output_text in tqdm(zip(input_texts, output_texts)):
                # 원본 텍스트를 바로 추가
                augmented_inputs.append(input_text)
                augmented_outputs.append(output_text)
                
                # 각 증강 방법별로 여러 번 증강
                for augmenter in self.augmenters:
                    try:
                        for _ in range(num_augments_per_method):
                            augmented = augmenter.augment(input_text)[0]
                            if isinstance(augmented, str) and '[unk]' not in augmented.lower():
                                # 중복 체크 없이 그냥 추가
                                augmented_inputs.append(augmented)
                                augmented_outputs.append(output_text)
                                augmented_count += 1
                    except Exception as e:
                        logging.warning(f"Augmentation failed: {str(e)}, {augmenter}")
                        continue

            logging.info(f"Original samples: {len(input_texts)}")
            logging.info(f"New augmentations: {augmented_count}")
            logging.info(f"Total augmented samples: {len(augmented_inputs)}")
            
            return augmented_inputs, augmented_outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')

    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    
    raw_data_path = os.path.join(project_root, 'ai', 'data', 'raw', 'simplified_generated_dataset.json')
    augmented_data_path = os.path.join(project_root, 'ai', 'data', 'augmented', 'simplified_augmented_dataset.json')
    
    logging.info(f"Looking for raw data at: {raw_data_path}")

    try:
        with open(raw_data_path, 'r') as f:
            data = json.load(f)
            input_texts = [item['input'] for item in data['dataset']]
            output_texts = [item['output'] for item in data['dataset']]
            
        logging.info(f"Loaded {len(input_texts)} input-output pairs for augmentation.")
        
        # Create augmenter and run augmentation
        augmenter = QueryAugmenterNlpAug()
        augmented_inputs, augmented_outputs = augmenter.augment(input_texts, output_texts)
        
        # Save results
        os.makedirs(os.path.dirname(augmented_data_path), exist_ok=True)
        with open(augmented_data_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": [
                    {"input": inp, "output": out} 
                    for inp, out in zip(augmented_inputs, augmented_outputs)
                ]
            }, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved {len(augmented_inputs)} augmented samples to {augmented_data_path}")
        
    except Exception as e:
        logging.error(f"Error during augmentation process: {e}")
        sys.exit(1)