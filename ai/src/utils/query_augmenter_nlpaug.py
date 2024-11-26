import logging
import nlpaug.augmenter.word as naw
from typing import List, Tuple, Set
import nltk
import re
import random

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

        # 숫자 표현 정의
        self.number_variations = {
            'cardinal': {
                1: ['one', 'single', '1'],
                2: ['two', 'couple', '2'],
                3: ['three', '3'],
                4: ['four', '4'],
                5: ['five', '5'],
                10: ['ten', '10']
            },
            'ordinal': {
                1: ['first', '1st'],
                2: ['second', '2nd'],
                3: ['third', '3rd'],
                4: ['fourth', '4th'],
                5: ['fifth', '5th']
            }
        }

        # 템플릿 구성요소 정의
        self.template_components = {
            'action_verbs': ['show', 'get', 'display', 'find'],
            'quantity_words': ['last', 'recent', 'latest', 'previous', 'past'],
            'transaction_terms': ['transactions', 'transaction records', 'transaction history', 'entries']
        }

        # 쿼리 템플릿 정의
        self.query_templates = [
            "{action} {number} {trans}",                          # show three transactions
            "{action} {quantity} {number} {trans}",              # show last three transactions
            "{action} me {number} {trans}",                      # show me three transactions
            "{action} me {quantity} {number} {trans}",           # show me last three transactions
            "I want to {action} {number} {trans}",               # I want to see three transactions
            "I need {number} {trans}",                           # I need three transactions
            "Could you {action} {number} {trans}",               # Could you show three transactions
            "Please {action} {number} {trans}"                   # Please show three transactions
        ]

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

        # 숫자 패턴 추가
        number_words = '|'.join(list(self.number_variations['cardinal'].values())[0] + 
                              [item for sublist in self.number_variations['cardinal'].values() 
                               for item in sublist])
        self.number_pattern = re.compile(f'\\b({number_words}|\\d+)\\s+{"|".join(self.template_components["transaction_terms"])}\\b', 
                                       re.IGNORECASE)
        
        self.CODE_PREFIX = "print(TransactionFilter(data)"

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
        
    def _generate_random_hex(self, length: int = 130) -> str:
        """130자리 랜덤 16진수 문자열 생성"""
        hex_chars = '01234567889abcdef'
        return ''.join(random.choice(hex_chars) for _ in range(length))
    
    def _replace_hex_values(self, text: str) -> Tuple[str, dict]:
        """텍스트 내의 130자리 16진수를 새로운 랜덤값으로 대체하고 매핑 정보 반환"""
        new_text = text
        hex_mapping = {}  # 원본 hex와 새로운 hex 간의 매핑 저장
        
        # from/to 패턴 확인
        from_match = self.from_pattern.search(text)
        to_match = self.to_pattern.search(text)
        
        # src_pk/pk 패턴 확인
        src_pk_match = self.src_pk_pattern.search(text) or self.src_pk_pattern2.search(text)
        pk_match = self.pk_pattern.search(text) or self.pk_pattern2.search(text)
        
        # 모든 16진수 값 찾기
        hex_values = self.hex_pattern.findall(text)
        
        for hex_value in hex_values:
            if hex_value not in hex_mapping:
                new_hex = self._generate_random_hex()
                hex_mapping[hex_value] = new_hex
                new_text = new_text.replace(hex_value, new_hex)
        
        # from/to나 src_pk/pk 패턴이 있는 경우 관계 유지를 위한 로깅
        if (from_match and to_match) or (src_pk_match and pk_match):
            logging.info(f"Found paired hex values. Mapping: {hex_mapping}")
        
        return new_text, hex_mapping


    def _clean_output_code(self, code: str) -> str:
        """모든 종류의 공백 제거"""
        return ''.join(code.split())
    
    def _wrap_value(self, value: str) -> str:
        """pk, timestamp 값을 작은 따옴표로 감싸는 함수"""
        return f"('{value}')"
    
    def _extract_transaction_count(self, text: str) -> int:
        """텍스트에서 트랜잭션 개수 추출"""
        match = self.number_pattern.search(text.lower())
        if match:
            number = match.group(1)
            # 숫자가 문자로 되어있으면 변환
            for num, variations in self.number_variations['cardinal'].items():
                if number.lower() in [var.lower() for var in variations]:
                    return num
            # 숫자 문자열이면 정수로 변환
            try:
                return int(number)
            except ValueError:
                pass
        return None

    def _analyze_hex_values(self, text: str) -> dict:
        """필요한 정보 분석"""
        result = {
            'src_pk': None,
            'pk': None,
            'timestamp': None,
            'func_name': None,
            'transaction_count': self._extract_transaction_count(text)  # 트랜잭션 개수 추가
        }
        
        # 함수명 검사 (setup, on, off)
        func_match = self.func_name_pattern.search(text.lower())
        if func_match:
            result['func_name'] = func_match.group(1).lower()

        # 패턴 존재 여부 확인
        has_from = 'from' in text.lower()
        has_to = 'to' in text.lower()
        has_pk_pattern = 'pk' in text.lower() or 'src_pk' in text.lower()
        
        # timestamp 확인
        timestamp_match = self.timestamp_pattern.search(text)
        if timestamp_match:
            result['timestamp'] = timestamp_match.group(0)
        
        # get_result 패턴 확인
        get_result_pattern = re.compile(r"get_result\s*\(\s*'?([0-9a-fA-F]{130})'?\s*\)")
        get_result_match = get_result_pattern.search(text)
        if get_result_match:
            result['pk'] = get_result_match.group(1)
            return result

        # 1. src_pk 직접 패턴 확인
        src_pk_match = self.src_pk_pattern.search(text) or self.src_pk_pattern2.search(text)
        if src_pk_match:
            result['src_pk'] = src_pk_match.group(1)
        
        # 2. pk 직접 패턴 확인
        pk_match = self.pk_pattern.search(text) or self.pk_pattern2.search(text)
        if pk_match:
            result['pk'] = pk_match.group(1)
        
        # 3. from 패턴 확인
        if not result['src_pk']:
            from_match = self.from_pattern.search(text)
            if from_match:
                result['src_pk'] = from_match.group(1)
                if not result['pk']:
                    remaining_hex = [h for h in self.hex_pattern.findall(text) 
                                if h != result['src_pk'] and 
                                text.index(h) > text.index(from_match.group(1))]
                    if remaining_hex:
                        result['pk'] = remaining_hex[0]
        
        # 4. to 패턴 확인
        if not result['pk']:
            to_match = self.to_pattern.search(text)
            if to_match:
                result['pk'] = to_match.group(1)
                if not result['src_pk']:
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
            if val is not None:
                keywords.add(str(val))
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
        """분석된 값을 기반으로 공백없는 필터 코드 생성"""
        filter_parts = []
        
        if values['func_name']:
            filter_parts.append(f"by_func_name{self._wrap_value(values['func_name'])}")
        if values['src_pk']:
            filter_parts.append(f"by_src_pk{self._wrap_value(values['src_pk'])}")
        if values['pk']:
            filter_parts.append(f"by_pk{self._wrap_value(values['pk'])}")
        if values['timestamp']:
            filter_parts.append(f"by_timestamp{self._wrap_value(values['timestamp'])}")
            
        if filter_parts:
            filter_chain = '.'.join(filter_parts) + '.get_result()'
            
            if values['transaction_count']:
                code = f"{self.CODE_PREFIX}.{filter_chain}[:{values['transaction_count']}])"
            else:
                code = f"{self.CODE_PREFIX}.{filter_chain})"
                
            return self._clean_output_code(code)
        
        return None

    def _create_number_specific_variations(self, text: str) -> List[str]:
        """템플릿 기반 변형 생성 - 항상 랜덤 pk 사용하기"""
        variations = set()  # 중복 방지를 위해 set 사용
        count = self._extract_transaction_count(text)
        
        if count and count in self.number_variations['cardinal']:
            number_forms = self.number_variations['cardinal'][count]
            
            # 각 템플릿에 대해
            for template in self.query_templates:
                # 각 구성요소의 가능한 조합을 적용
                for action in self.template_components['action_verbs']:
                    for number in number_forms:
                        for trans in self.template_components['transaction_terms']:
                            # 기본 템플릿 (quantity가 없는 경우)
                            if "{quantity}" not in template:
                                variation = template.format(
                                    action=action,
                                    number=number,
                                    trans=trans
                                )
                                variations.add(variation)
                            # quantity가 있는 템플릿
                            else:
                                for quantity in self.template_components['quantity_words']:
                                    variation = template.format(
                                        action=action,
                                        quantity=quantity,
                                        number=number,
                                        trans=trans
                                    )
                                    variations.add(variation)
        
        # PK 값 추가
        final_variations = []
        for variation in variations:
        # to/from 랜덤 선택하고 랜덤 PK 추가
            for _ in range(2):  # 각 변형에 대해 to와 from 버전 모두 생성
                if random.choice(['to', 'from']) == 'to':
                    final_variations.append(f"{variation} to {self._generate_random_hex()}")
                else:
                    final_variations.append(f"{variation} from {self._generate_random_hex()}")
        
        return final_variations
    
    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2) -> Tuple[List[str], List[str]]:
        augmented_inputs = []
        augmented_outputs = []
        
        # 원본 데이터도 새로운 hex 값으로 변경
        for input_text, output_code in zip(input_texts, output_texts):
            # 입력 텍스트의 hex 값 분석
            values = self._analyze_hex_values(input_text)
            new_values = values.copy()
            
            # 새로운 hex 값 생성 및 적용
            new_input = input_text
            new_output = output_code
            
            if values['src_pk']:
                new_hex = self._generate_random_hex()
                new_input = new_input.replace(values['src_pk'], new_hex)
                new_output = new_output.replace(values['src_pk'], new_hex)
                new_values['src_pk'] = new_hex
                
            if values['pk']:
                new_hex = self._generate_random_hex()
                new_input = new_input.replace(values['pk'], new_hex)
                new_output = new_output.replace(values['pk'], new_hex)
                new_values['pk'] = new_hex
            
            augmented_inputs.append(new_input)
            augmented_outputs.append(new_output)
        
        for idx, (input_text, output_code) in enumerate(zip(input_texts, output_texts)):
            try:
                logging.info(f"\n처리 중인 입력 텍스트: {input_text}")
                
                # 입력 텍스트의 hex 값 분석
                values = self._analyze_hex_values(input_text)
                
                # 1. 템플릿 기반 변형 생성
                template_variations = self._create_number_specific_variations(input_text)
                if template_variations:
                    for template in template_variations:
                        # 각 템플릿마다 새로운 hex 값 생성
                        new_values = values.copy()
                        new_template = template
                        new_output = output_code
                        
                        if values['src_pk']:
                            new_hex = self._generate_random_hex()
                            new_template = new_template.replace(values['src_pk'], new_hex)
                            new_output = new_output.replace(values['src_pk'], new_hex)
                            new_values['src_pk'] = new_hex
                            
                        if values['pk']:
                            new_hex = self._generate_random_hex()
                            new_template = new_template.replace(values['pk'], new_hex)
                            new_output = new_output.replace(values['pk'], new_hex)
                            new_values['pk'] = new_hex
                        
                        augmented_inputs.append(new_template)
                        augmented_outputs.append(new_output)
                
                # 키워드 추출 (hex 값 제외)
                keywords = self._get_keywords(input_text) - set(self.hex_pattern.findall(input_text))
                logging.info(f"보존할 키워드: {keywords}")
                
                nlp_variations = set()
                
                # 2. 동의어 기반 변형
                try:
                    syn_texts = self.aug_syn.augment(input_text, n=num_variations)
                    logging.info(f"동의어 교체 결과: {syn_texts}")
                    if isinstance(syn_texts, list):
                        for text in syn_texts:
                            # 새로운 hex 값 생성 및 적용
                            new_values = values.copy()
                            preserved = self._preserve_keywords(text[0] if isinstance(text, list) else text, keywords)
                            new_text = preserved
                            new_output = output_code
                            
                            if values['src_pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['src_pk'], new_hex)
                                new_output = new_output.replace(values['src_pk'], new_hex)
                                
                            if values['pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['pk'], new_hex)
                                new_output = new_output.replace(values['pk'], new_hex)
                                
                            nlp_variations.add((new_text, new_output))
                except Exception as e:
                    logging.error(f"동의어 교체 중 에러: {str(e)}")
                
                # 3. 문맥 기반 삽입
                try:
                    insert_texts = self.aug_insert.augment(input_text, n=num_variations)
                    logging.info(f"단어 삽입 결과: {insert_texts}")
                    if isinstance(insert_texts, list):
                        for text in insert_texts:
                            # 새로운 hex 값 생성 및 적용
                            new_values = values.copy()
                            preserved = self._preserve_keywords(text[0] if isinstance(text, list) else text, keywords)
                            new_text = preserved
                            new_output = output_code
                            
                            if values['src_pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['src_pk'], new_hex)
                                new_output = new_output.replace(values['src_pk'], new_hex)
                                
                            if values['pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['pk'], new_hex)
                                new_output = new_output.replace(values['pk'], new_hex)
                                
                            nlp_variations.add((new_text, new_output))
                except Exception as e:
                    logging.error(f"단어 삽입 중 에러: {str(e)}")
                
                # 4. BERT 기반 대체 (동일한 패턴 적용)
                try:
                    context_texts = self.aug_word_embs.augment(input_text, n=num_variations)
                    logging.info(f"BERT 문맥 변경 결과: {context_texts}")
                    if isinstance(context_texts, list):
                        for text in context_texts:
                            new_values = values.copy()
                            preserved = self._preserve_keywords(text[0] if isinstance(text, list) else text, keywords)
                            new_text = preserved
                            new_output = output_code
                            
                            if values['src_pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['src_pk'], new_hex)
                                new_output = new_output.replace(values['src_pk'], new_hex)
                                
                            if values['pk']:
                                new_hex = self._generate_random_hex()
                                new_text = new_text.replace(values['pk'], new_hex)
                                new_output = new_output.replace(values['pk'], new_hex)
                                
                            nlp_variations.add((new_text, new_output))
                except Exception as e:
                    logging.error(f"BERT 문맥 변경 중 에러: {str(e)}")
                
                logging.info(f"생성된 모든 NLP 변형: {nlp_variations}")
                
                # NLP 변형 검증 및 필터링
                for variation, var_output in nlp_variations:
                    if (variation != input_text and  # 원본과 동일하지 않고
                        self._check_valid_variation(variation, var_output)):  # 유효한 변형인 경우
                        augmented_inputs.append(variation)
                        augmented_outputs.append(var_output)
                
                if idx % 10 == 0:
                    logging.info(f"Processed {idx+1}/{len(input_texts)} items")
                    logging.info(f"Current variations for item {idx}: {len(nlp_variations)}")
                
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