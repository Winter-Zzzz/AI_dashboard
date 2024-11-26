import logging
import nlpaug.augmenter.word as naw
from typing import List, Tuple, Set
import nltk
import re

def download_nltk_data_if_needed():
    nltk_packages = ['wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4', 'punkt']
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            logging.info(f"Package {package} already exists. Skipping download.")
        except LookupError:
            try:
                nltk.download(package)
                logging.info(f"Successfully downloaded {package}")
            except Exception as e:
                logging.warning(f"NLTK 데이터 '{package}' 다운로드 중 경고: {str(e)}")

class QueryAugmenterNlpAug:
    def __init__(self):
        # NLTK 데이터 다운로드 체크 및 필요시 다운로드
        download_nltk_data_if_needed()

        # 나머지 코드는 동일...

class QueryAugmenterNlpAug:
    def __init__(self):
        # NLTK 데이터 다운로드
        download_nltk_data_if_needed()

        # 트랜잭션 관련 키워드 정의
        self.preserved_keywords = {
            'transaction', 'transactions', 'source', 'destination', 'address',
            'timestamp', 'function', 'calls', 'setup', 'on', 'off',
            'show', 'find', 'display', 'get',
            'latest', 'recent', 'earliest'
        }

        try:
            # NlpAug 증강기 초기화 - 보수적인 파라미터 설정
            self.aug_syn = naw.SynonymAug(
                aug_src='wordnet',
                aug_p=0.3,  # 더 보수적인 확률
                aug_min=1,
                stopwords=self.preserved_keywords
            )
            
            self.aug_insert = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', 
                action="insert",
                aug_p=0.3,
                aug_min=1,
                stopwords=self.preserved_keywords
            )
            
            self.aug_word_embs = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                action="substitute",
                aug_p=0.3,
                aug_min=1,
                stopwords=self.preserved_keywords
            )
            
            logging.info("NlpAug 증강기 초기화 완료")
            
        except Exception as e:
            logging.error(f"증강기 초기화 중 에러 발생: {str(e)}")
            raise e

    def _get_keywords(self, text: str) -> Set[str]:
        """보존할 키워드 추출 (128자리 16진수 PK 처리)"""
        keywords = set()
        words = text.split()
        
        # 130자리 16진수 형태의 src_pk, pk 값 보존
        for word in words:
            if len(word) == 130 and all(c in '0123456789abcdefABCDEF' for c in word):
                keywords.add(word)
            elif len(word) == 10 and word.isdigit():
                keywords.add(word)
                
        return keywords

    def _preserve_keywords(self, text: str, keywords: Set[str]) -> str:
        """키워드 보존"""
        result = text
        words = result.split()
        for keyword in keywords:
            if keyword not in result and words:
                result = result.replace(words[0], keyword, 1)
        return result

    def _check_valid_variation(self, variation: str, output_code: str) -> bool:
        """변형된 텍스트가 유효한지 확인"""
        # output 코드에서 src_pk와 pk 값 추출
        src_pk_val = output_code.split("by_src_pk('")[1].split("'")[0] if "by_src_pk('" in output_code else None
        pk_val = output_code.split("by_pk('")[1].split("'")[0] if "by_pk('" in output_code else None
        
        # src_pk/pk 값이 변형된 텍스트에도 존재하는지 확인
        if src_pk_val and src_pk_val not in variation:
            return False
        if pk_val and pk_val not in variation:
            return False
            
        return True

    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2) -> Tuple[List[str], List[str]]:
        """NlpAug 기반 텍스트 증강"""
        augmented_inputs = []
        augmented_outputs = []
        
        # 원본 데이터 유지
        augmented_inputs.extend(input_texts)
        augmented_outputs.extend(output_texts)
        
        for idx, (input_text, output_code) in enumerate(zip(input_texts, output_texts)):
            try:
                logging.info(f"\n처리 중인 입력 텍스트: {input_text}")
                
                # 보존할 키워드 추출
                keywords = self._get_keywords(input_text)
                logging.info(f"보존할 키워드: {keywords}")
                
                variations = set()
                
                # 1. 동의어 기반 변형
                try:
                    syn_texts = self.aug_syn.augment(input_text, n=num_variations)
                    logging.info(f"동의어 교체 결과: {syn_texts}")
                    if isinstance(syn_texts, list):
                        for text in syn_texts:
                            if isinstance(text, str):
                                preserved = self._preserve_keywords(text, keywords)
                            else:
                                preserved = self._preserve_keywords(text[0], keywords)
                            variations.add(preserved)
                except Exception as e:
                    logging.error(f"동의어 교체 중 에러: {str(e)}")
                
                # 2. 문맥 기반 삽입
                try:
                    insert_texts = self.aug_insert.augment(input_text, n=num_variations)
                    logging.info(f"단어 삽입 결과: {insert_texts}")
                    if isinstance(insert_texts, list):
                        for text in insert_texts:
                            if isinstance(text, str):
                                preserved = self._preserve_keywords(text, keywords)
                            else:
                                preserved = self._preserve_keywords(text[0], keywords)
                            variations.add(preserved)
                except Exception as e:
                    logging.error(f"단어 삽입 중 에러: {str(e)}")
                
                # 3. BERT 기반 대체
                try:
                    context_texts = self.aug_word_embs.augment(input_text, n=num_variations)
                    logging.info(f"BERT 문맥 변경 결과: {context_texts}")
                    if isinstance(context_texts, list):
                        for text in context_texts:
                            if isinstance(text, str):
                                preserved = self._preserve_keywords(text, keywords)
                            else:
                                preserved = self._preserve_keywords(text[0], keywords)
                            variations.add(preserved)
                except Exception as e:
                    logging.error(f"BERT 문맥 변경 중 에러: {str(e)}")
                
                logging.info(f"생성된 모든 변형: {variations}")
                
                # 변형 검증 및 필터링
                valid_variations = set()
                for variation in variations:
                    if (variation != input_text and  # 원본과 동일하지 않고
                        self._check_valid_variation(variation, output_code)):  # 유효한 변형인 경우
                        valid_variations.add(variation)
                
                logging.info(f"유효한 변형 수: {len(valid_variations)}")
                if valid_variations:
                    logging.info(f"유효한 변형들: {valid_variations}")
                    augmented_inputs.extend(valid_variations)
                    augmented_outputs.extend([output_code] * len(valid_variations))
                
                if idx % 10 == 0:
                    logging.info(f"Processed {idx+1}/{len(input_texts)} items")
                    logging.info(f"Current variations for item {idx}: {len(variations)}")
                
            except Exception as e:
                logging.error(f"증강 중 에러 발생 (idx={idx}): {str(e)}")
                continue
        
        # 중복 제거
        unique_pairs = list(dict.fromkeys(zip(augmented_inputs, augmented_outputs)))
        augmented_inputs, augmented_outputs = zip(*unique_pairs)
        
        logging.info(f"\n=== 최종 증강 결과 ===")
        logging.info(f"원본 데이터 크기: {len(input_texts)}")
        logging.info(f"증강 후 크기: {len(augmented_inputs)}")
        if len(augmented_inputs) > len(input_texts):
            logging.info(f"증강 예시:")
            for i, text in enumerate(list(set(augmented_inputs) - set(input_texts))[:3], 1):
                logging.info(f"예시 {i}: {text}")
        
        return list(augmented_inputs), list(augmented_outputs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    augmenter = QueryAugmenterNlpAug()
    
    # 테스트 데이터
    test_inputs = [
        "Show 3 transactions to 04750bfae2e57e7160cb5ead399ab37afdb4a1451a0b96b08764296dbe8490d946f1312034836474ccf7070b44d3e98f03dca538d148aff42fce155f58243de60d"
    ]

    test_outputs = [
        "print(TransactionFilter(data).by_pk('04750bfae2e57e7160cb5ead399ab37afdb4a1451a0b96b08764296dbe8490d946f1312034836474ccf7070b44d3e98f03dca538d148aff42fce155f58243de60d').get_result()[:3])",
        # "print(TransactionFilter(data).by_src_pk('137e79e9ffe13c164580d4cb388f70f6cee4dcc5eafc0c1ddf0296df00d885e47bbd3a1d22c17c61dc663a05030cd7db0f3f114590da4b0ee2d5679906affa54').sort(reverse=True).get_result()[:5])"
    ]
    
    augmented_inputs, augmented_outputs = augmenter.augment(test_inputs, test_outputs)