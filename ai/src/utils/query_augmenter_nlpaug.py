import logging
import nlpaug.augmenter.word as naw
from typing import List, Tuple, Set
import nltk

# NLTK 필요 데이터 다운로드
try:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('omw-1.4')
    nltk.download('punkt')
except Exception as e:
    logging.warning(f"NLTK 데이터 다운로드 중 경고: {str(e)}")

class QueryAugmenterNlpAug:
    def __init__(self):
        try:
            # NlpAug 증강기 초기화 - 파라미터 조정
            self.aug_syn = naw.SynonymAug(
                aug_src='wordnet',
                aug_p=0.5,  # 증가
                aug_min=1
            )
            
            self.aug_insert = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', 
                action="insert",
                aug_p=0.5,  # 추가
                aug_min=1   # 최소 1개 변경
            )
            
            self.aug_word_embs = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                action="substitute",
                aug_p=0.5,  # 증가
                aug_min=1   # 최소 1개 변경
            )
            
            logging.info("NlpAug 증강기 초기화 완료")
            
        except Exception as e:
            logging.error(f"NlpAug 증강기 초기화 중 에러 발생: {str(e)}")
            raise e

    def _get_keywords(self, text: str) -> Set[str]:
        """보존할 키워드 추출 (128자리 16진수 PK 처리)"""
        keywords = set()
        words = text.split()
        
        # 128자리 16진수 형태의 src_pk, pk 값 보존
        for word in words:
            if len(word) == 128 and all(c in '0123456789abcdefABCDEF' for c in word):
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
        src_pk_val = output_code.split("src_pk='")[1].split("'")[0] if "src_pk='" in output_code else None
        pk_val = output_code.split("pk='")[1].split("'")[0] if "pk='" in output_code else None
        
        # src_pk/pk 값이 변형된 텍스트에도 존재하는지 확인
        if src_pk_val and src_pk_val not in variation:
            return False
        if pk_val and pk_val not in variation:
            return False
            
        return True

    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 20) -> Tuple[List[str], List[str]]:
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
                
                # 1. 동의어 교체로 증강
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
                
                # 2. 단어 삽입으로 증강
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
                
                # 3. BERT 기반 문맥 변경으로 증강
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
                
                # 변형 결과 검증 및 추가
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
        
        # 중복 제거 # !!!! 중복을 제거하여 최종 결과를 고유한 쌍으로 만듦
        unique_pairs = list(dict.fromkeys(zip(augmented_inputs, augmented_outputs)))
        augmented_inputs, augmented_outputs = zip(*unique_pairs)
        
        logging.info(f"\n=== 최종 증강 결과 ===")
        logging.info(f"원본 데이터 크기: {len(input_texts)}")
        logging.info(f"증강 후 크기: {len(augmented_inputs)}")
        if len(augmented_inputs) > len(input_texts):
            logging.info(f"증강 예시:")
            for i, text in enumerate(list(set(augmented_inputs) - set(input_texts))[:3], 1):
                logging.info(f"예시 {i}: {text}")
        else:
            logging.warning("증강 데이터가 생성되지 않았거나 모두 필터링되었습니다.")
        
        return list(augmented_inputs), list(augmented_outputs)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    augmenter = QueryAugmenterNlpAug()
    test_inputs = [
        "Query 5 latest device responses from 1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef to 1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "Show all status updates from 1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    ]
    test_outputs = [
        "const src_pk='1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef';const pk='1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef';const result=data.filter(item=>item.pk===pk).flatMap(item=>item.transactions).filter(tx=>tx.src_pk===src_pk).sort((a,b)=>parseInt(b.timestamp)-parseInt(a.timestamp)).slice(0,5);console.log(result);",
        "const src_pk='1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef';const pk='1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef';const result=data.filter(item=>item.pk===pk).flatMap(item=>item.transactions).filter(tx=>tx.src_pk===src_pk).sort((a,b)=>parseInt(b.timestamp)-parseInt(a.timestamp)).slice(0,5);console.log(result);"
    ]
    
    augmented_inputs, augmented_outputs = augmenter.augment(test_inputs, test_outputs)
    logging.info(f"증강된 입력 텍스트: {augmented_inputs}")
    logging.info(f"증강된 출력 코드: {augmented_outputs}")
