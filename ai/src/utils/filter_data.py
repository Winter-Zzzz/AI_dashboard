import json
from datetime import datetime
import os
import sys



# JSON 파일 읽기 함수
def load_transactions_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

from datetime import datetime
from typing import List, Dict, Callable, Any, Optional

class TransactionFilter:
    def __init__(self, transactions: List[Dict]):
        self.transactions = transactions
        self._filtered_data = transactions

    def reset(self) -> 'TransactionFilter':
        self._filtered_data = self.transactions
        return self

    def by_pk(self, pk_value: int | str = -1) -> 'TransactionFilter':
        if pk_value != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data 
                if transaction['pk'] == pk_value
            ]
        return self

    def by_src_pk(self, src_pk_value: int | str = -1) -> 'TransactionFilter':
        if src_pk_value != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                if transaction.get('src_pk') == src_pk_value
            ]
        return self

    def by_func_name(self, func_name: int | str = -1) -> 'TransactionFilter':
        if func_name != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                if transaction.get('func_name') == func_name
            ]
        return self

    def by_timestamp(self, timestamp_value: int | str = -1, compare_func: Callable = lambda x, y: x > y) -> 'TransactionFilter':
        if timestamp_value != -1:
            timestamp_datetime = datetime.fromtimestamp(int(timestamp_value))
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                if compare_func(
                    datetime.fromtimestamp(int(transaction['timestamp'])),
                    timestamp_datetime
                )
            ]
        return self

    def sort(self, 
        key_func: Optional[Callable[[Dict], Any]] = None,
        order: int = 0) -> 'TransactionFilter':
        """
        결과 정렬
        Args:
            key_func: 정렬 키 함수. None이면 기본 정렬 함수 사용
            order: 정렬 순서 (0: 오름차순, 1: 내림차순)
        Returns:
            TransactionFilter 인스턴스
        """
        try:
            if not self._filtered_data:  # 데이터가 비어있으면 그대로 반환
                return self
                
            # 첫 번째 항목의 키를 검사하여 정렬 키 결정
            first_item = self._filtered_data[0]
            
            def default_key(x: Dict) -> Any:
                # timestamp 키가 있는지 확인
                if 'timestamp' in x:
                    return x['timestamp']
                # transactions 키가 있고 그 안에 timestamp가 있는지 확인
                elif 'transactions' in x and x['transactions']:
                    return x['transactions'][0]['timestamp']
                # 둘 다 없으면 0 반환
                return 0
            
            if isinstance(key_func, (int, float)):
                order = key_func
                actual_key = default_key
            else:
                actual_key = default_key if key_func in [-1, None] else key_func
            
            reverse = bool(order == 1)
            self._filtered_data = sorted(self._filtered_data, key=actual_key, reverse=reverse)
            return self
        
        except Exception as e:
            print(f"정렬 중 에러 발생: {str(e)}")
            return self
    
    def get_result(self, count: int = -1) -> List[Dict]:
        if count != -1:
            return self._filtered_data[:count]
        return self._filtered_data
    
def load_json_data(file_path):
    """JSON 파일에서 데이터 로드"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"데이터 로드 완료: {file_path}")
        return data
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {str(e)}")
        return None


# 테스트를 위한 필터링 코드
test_data = [
    {'pk': '04750bfae2e57e7160cb5ead399ab37afdb4a1451a0b96b08764296dbe8490d946f1312034836474ccf7070b44d3e98f03dca538d148aff42fce155f58243de60d', 'func_name': 'world', 'timestamp': 1732446169},
    {'pk': '04750bfae2e57e7160cb5ead399ab37afdb4a1451a0b96b08764296dbe8490d946f1312034836474ccf7070b44d3e98f03dca538d148aff42fce155f58243de60d', 'func_name': 'feedback', 'timestamp': 1732255734}
]

filter = TransactionFilter(test_data).reset()\
    .by_pk('04750bfae2e57e7160cb5ead399ab37afdb4a1451a0b96b08764296dbe8490d946f1312034836474ccf7070b44d3e98f03dca538d148aff42fce155f58243de60d')\
    .sort(1)\
    .get_result()

print(filter)