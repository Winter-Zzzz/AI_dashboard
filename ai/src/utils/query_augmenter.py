import random
import re
from typing import List, Tuple, Optional
import logging

class QueryAugmenter:
    def __init__(self):
        self.query_templates = {
            'src_dest': [
                "Find {count} latest interactions from {src} to {dest}",
                "Get {count} most recent records between {src} and {dest}",
                "Show {count} transactions from {src} to {dest}",
                "Retrieve {count} latest events from {src} to {dest}",
                "Display {count} recent messages {src} sent to {dest}"
            ],
            'single_target': [
                "Find all records from {target}",
                "Get complete history of {target}",
                "Show all events related to {target}",
                "Retrieve full log from {target}",
                "Display all messages by {target}"
            ]
        }
    
    def extract_numbers_from_query(self, query: str) -> List[str]:
        """쿼리에서 숫자들을 추출"""
        return re.findall(r'\d+', query)
    
    def is_src_dest_query(self, query: str) -> bool:
        """src와 dest가 있는 쿼리인지 확인"""
        return "to" in query.lower() or "from" in query.lower() or "between" in query.lower()
    
    def extract_src_dest(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """쿼리에서 src와 dest를 추출"""
        parts = query.lower().split()
        src = dest = None
        
        try:
            # 'to', 'from', 'between' 키워드로 src/dest 위치 파악
            for i, part in enumerate(parts):
                if part == 'from' and i + 1 < len(parts):
                    src = parts[i + 1]
                elif part == 'to' and i + 1 < len(parts):
                    dest = parts[i + 1]
                elif part == 'between' and i + 2 < len(parts) and parts[i + 2] == 'and':
                    src = parts[i + 1]
                    dest = parts[i + 3]
        except Exception as e:
            logging.warning(f"Failed to extract src/dest from query: {query}, Error: {str(e)}")
            
        return src, dest

    def extract_single_target(self, query: str) -> Optional[str]:
        """단일 대상 쿼리에서 target을 추출"""
        parts = query.lower().split()
        try:
            for i, part in enumerate(parts):
                if part in ['by', 'from', 'to'] and i + 1 < len(parts):
                    return parts[i + 1]
        except Exception as e:
            logging.warning(f"Failed to extract target from query: {query}, Error: {str(e)}")
        return None

    def augment_src_dest_query(self, query: str, count: str, src: str, dest: str, num_variations: int) -> List[str]:
        """src/dest 쿼리에 대한 변형 생성"""
        variations = []
        try:
            templates = random.sample(
                self.query_templates['src_dest'], 
                min(num_variations, len(self.query_templates['src_dest']))
            )
            
            for template in templates:
                variation = template.format(count=count, src=src, dest=dest)
                variations.append(variation)
        except Exception as e:
            logging.warning(f"Failed to augment src/dest query: {query}, Error: {str(e)}")
        
        return variations

    def augment_single_target_query(self, query: str, target: str, num_variations: int) -> List[str]:
        """단일 대상 쿼리에 대한 변형 생성"""
        variations = []
        try:
            templates = random.sample(
                self.query_templates['single_target'], 
                min(num_variations, len(self.query_templates['single_target']))
            )
            
            for template in templates:
                variation = template.format(target=target)
                variations.append(variation)
        except Exception as e:
            logging.warning(f"Failed to augment single target query: {query}, Error: {str(e)}")
        
        return variations

    def augment(self, input_texts: List[str], output_texts: List[str], num_variations: int = 2) -> Tuple[List[str], List[str]]:
        """쿼리 데이터 증강 메인 함수"""
        augmented_inputs = []
        augmented_outputs = []
        
        for inp, out in zip(input_texts, output_texts):
            # 원본 데이터 추가
            augmented_inputs.append(inp)
            augmented_outputs.append(out)
            
            try:
                numbers = self.extract_numbers_from_query(inp)
                if not numbers:
                    continue
                
                if self.is_src_dest_query(inp):
                    src, dest = self.extract_src_dest(inp)
                    if src and dest:
                        variations = self.augment_src_dest_query(
                            query=inp,
                            count=numbers[0],
                            src=src,
                            dest=dest,
                            num_variations=num_variations
                        )
                        for var in variations:
                            augmented_inputs.append(var)
                            augmented_outputs.append(out)
                else:
                    target = self.extract_single_target(inp)
                    if target:
                        variations = self.augment_single_target_query(
                            query=inp,
                            target=target,
                            num_variations=num_variations
                        )
                        for var in variations:
                            augmented_inputs.append(var)
                            augmented_outputs.append(out)
                            
            except Exception as e:
                logging.error(f"Error during augmentation for query: {inp}, Error: {str(e)}")
                continue
        
        # 중복 제거
        unique_pairs = list(dict.fromkeys(zip(augmented_inputs, augmented_outputs)))
        augmented_inputs, augmented_outputs = zip(*unique_pairs)
        
        return list(augmented_inputs), list(augmented_outputs)

# 템플릿 추가/수정을 위한 메서드
    def add_src_dest_template(self, template: str):
        """src/dest 쿼리 템플릿 추가"""
        if all(key in template for key in ['{count}', '{src}', '{dest}']):
            self.query_templates['src_dest'].append(template)
        else:
            raise ValueError("Template must contain {count}, {src}, and {dest} placeholders")

    def add_single_target_template(self, template: str):
        """단일 대상 쿼리 템플릿 추가"""
        if '{target}' in template:
            self.query_templates['single_target'].append(template)
        else:
            raise ValueError("Template must contain {target} placeholder")
        
if __name__ == "__main__":
    augmenter = QueryAugmenter()