import re
from typing import List

def preprocess_input_text(text: str) -> str:
    """
    입력 텍스트 전처리
    
    Args:
        text (str): 원본 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    # 기본적인 전처리
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'<pad>|</s>|<unk>|<extra_id_\d+>', '', text)
    return text

def preprocess_output_text(text: str) -> str:
    """
    출력 텍스트 전처리
    
    Args:
        text (str): 원본 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    # 특수 토큰 제거
    text = re.sub(r'<pad>|</s>|<unk>|<extra_id_\d+>', '', text)

    # 'function'을 제거하고 간단한 형태로 변환
    text = re.sub(r"'on function'", "'on'", text)
    text = re.sub(r"'off function'", "'off'", text)
    text = re.sub(r"'setup function'", "'setup'", text)

    return text.strip()