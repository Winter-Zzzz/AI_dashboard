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
        # NLTK 데이터 다운로드
        download_nltk_data_if_needed()

        # 트랜잭션 관련 키워드 정의
        self.preserved_keywords = {
            'transaction', 'transactions', 'timestamp', 'function', 'setup', 'on', 'off',
            'show', 'find', 'display', 'get', 'latest', 'recent', 'earliest', 'from', 'to', 'src_pk', 'pk'
        }

         # 모든 패턴 정규식 정의
        self.hex_pattern = re.compile(r'[0-9a-fA-F]{130}')
        self.timestamp_pattern = re.compile(r'\b\d{10}\b')
        
        # from/to 패턴
        self.from_pattern = re.compile(r'from\s+([0-9a-fA-F]{130})')
        self.to_pattern = re.compile(r'to\s+([0-9a-fA-F]{130})')

        # src_pk/pk 직접 패턴
        self.src_pk_pattern = re.compile(r'src_?pk\s*[=:]\s*([0-9a-fA-F]{130})')
        self.src_pk_pattern2 = re.compile(r'src_?pk\s+([0-9a-fA-F]{130})')
        self.pk_pattern = re.compile(r'(?<!src_)pk\s*[=:]\s*([0-9a-fA-F]{130})')
        self.pk_pattern2 = re.compile(r'(?<!src_)pk\s+([0-9a-fA-F]{130})')

        # 함수명 패턴 추가
        self.func_name_pattern = re.compile(r'\b(setup|on|off)\b', re.IGNORECASE)

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

    def _analyze_hex_values(self, text: str) -> dict:
        """모든 패턴에 따라 16진수 값 분석"""
        result = {
            'src_pk': None,
            'pk': None,
            'timestamp': None,
            'func_name': None 
        }
        
        # 함수명 검사 (setup, on, off)
        func_match = self.func_name_pattern.search(text.lower())
        if func_match:
            result['func_name'] = func_match.group(1).lower()

        # 패턴 존재 여부 확인
        has_from = 'from' in text.lower()
        has_to = 'to' in text.lower()
        has_pk_pattern = 'pk' in text.lower() or 'src_pk' in text.lower()
        
        # timestamp 확인 (항상 먼저 체크)
        timestamp_match = self.timestamp_pattern.search(text)
        if timestamp_match:
            result['timestamp'] = timestamp_match.group(0)
        
        # 1. src_pk 직접 패턴 확인
        src_pk_match = self.src_pk_pattern.search(text) or self.src_pk_pattern2.search(text)
        if src_pk_match:
            result['src_pk'] = src_pk_match.group(1)
        
        # 2. pk 직접 패턴 확인
        pk_match = self.pk_pattern.search(text) or self.pk_pattern2.search(text)
        if pk_match:
            result['pk'] = pk_match.group(1)
        
        # 3. from 패턴 확인
        if not result['src_pk']:  # src_pk가 아직 설정되지 않은 경우만
            from_match = self.from_pattern.search(text)
            if from_match:
                result['src_pk'] = from_match.group(1)
                # from 이후의 다른 16진수는 to가 없어도 pk로 처리
                if not result['pk']:  # pk가 아직 설정되지 않은 경우만
                    remaining_hex = [h for h in self.hex_pattern.findall(text) 
                                if h != result['src_pk'] and 
                                text.index(h) > text.index(from_match.group(1))]
                    if remaining_hex:
                        result['pk'] = remaining_hex[0]
        
        # 4. to 패턴 확인
        if not result['pk']:  # pk가 아직 설정되지 않은 경우만
            to_match = self.to_pattern.search(text)
            if to_match:
                result['pk'] = to_match.group(1)
                # to 이전의 다른 16진수는 from이 없어도 src_pk로 처리
                if not result['src_pk']:  # src_pk가 아직 설정되지 않은 경우만
                    remaining_hex = [h for h in self.hex_pattern.findall(text) 
                                if h != result['pk'] and 
                                text.index(h) < text.index(to_match.group(1))]
                    if remaining_hex:
                        result['src_pk'] = remaining_hex[0]
        
        # 5. 패턴이 없는 경우 - 첫 번째 16진수를 pk로 처리
        if not (has_from or has_to or has_pk_pattern):
            hex_values = self.hex_pattern.findall(text)
            if hex_values and not (result['src_pk'] or result['pk']):
                result['pk'] = hex_values[0]
            
        return result
    
    def _get_keywords(self, text: str) -> Set[str]:
        """보존할 키워드 추출 (128자리 16진수 PK 처리)"""
        keywords = set()
        values = self._analyze_hex_values(text)
        for val in values.values():
            if val:
                keywords.add(val)
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
        original_values = self._analyze_hex_values(output_code)
        variation_values = self._analyze_hex_values(variation)
        
        # 모든 값이 올바른 위치에 보존되었는지 확인
        if original_values['src_pk'] and original_values['src_pk'] != variation_values['src_pk']:
            return False
        if original_values['pk'] and original_values['pk'] != variation_values['pk']:
            return False
        if original_values['timestamp'] and original_values['timestamp'] != variation_values['timestamp']:
            return False
        if original_values['func_name'] and original_values['func_name'] != variation_values['func_name']:
            return False
            
        return True
    
    def _generate_filter_code(self, values: dict) -> str:
        """분석된 값을 기반으로 필터 코드 생성"""
        filter_parts = []
        
        # 필터 순서: src_pk -> pk -> timestamp -> get_result
        if values['func_name']:
            filter_parts.append(f"by_func_name('{values['func_name']}')")
        if values['src_pk']:
            filter_parts.append(f"by_src_pk('{values['src_pk']}')")
        if values['pk']:
            filter_parts.append(f"by_pk('{values['pk']}')")
        if values['timestamp']:
            filter_parts.append(f"by_timestamp('{values['timestamp']}')")
            
        if filter_parts:
            # 필터 체인 생성 후 get_result() 추가
            filter_chain = '.'.join(filter_parts) + '.get_result()'
            return f"print(TransactionFilter(data).{filter_chain})"
        return None


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
                
                # 입력 텍스트 분석
                values = self._analyze_hex_values(input_text)
                
                # 필터 코드 생성
                generated_code = self._generate_filter_code(values)
                if generated_code:
                    output_texts[idx] = generated_code
                
                # 키워드 추출
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