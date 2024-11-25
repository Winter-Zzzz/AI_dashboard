import json
from datetime import datetime



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
        """필터링을 초기 상태로 리셋"""
        self._filtered_data = self.transactions
        return self

    def by_pk(self, pk_value: str) -> 'TransactionFilter':
        """pk로 필터링"""
        self._filtered_data = [
            transaction for transaction in self._filtered_data 
            if transaction['pk'] == pk_value
        ]
        return self

    def by_src_pk(self, src_pk_value: str) -> 'TransactionFilter':
        """src_pk로 필터링"""
        self._filtered_data = [
            transaction for transaction in self._filtered_data
            for txn in transaction['transactions']
            if txn['src_pk'] == src_pk_value
        ]
        return self
    
    def by_func_name(self, func_name: str) -> 'TransactionFilter':
        """func_name으로 필터링"""
        self._filtered_data = [
            transaction for transaction in self._filtered_data
            for txn in transaction['transactions']
            if txn['func_name'] == func_name
        ]
        return self

    def by_timestamp(self, timestamp_value: str, compare_func: Callable = lambda x, y: x > y) -> 'TransactionFilter':
        """timestamp로 필터링. 비교 함수를 커스텀할 수 있음"""
        timestamp_datetime = datetime.fromtimestamp(int(timestamp_value))
        self._filtered_data = [
            transaction for transaction in self._filtered_data
            for txn in transaction['transactions']
            if compare_func(
                datetime.fromtimestamp(int(txn['timestamp'])),
                timestamp_datetime
            )
        ]
        return self

    def custom_filter(self, filter_func: Callable[[Dict], bool]) -> 'TransactionFilter':
        """커스텀 필터 함수 적용"""
        self._filtered_data = [
            transaction for transaction in self._filtered_data
            if filter_func(transaction)
        ]
        return self

    def sort(self, 
            key_func: Callable[[Dict], Any] = lambda x: x['transactions'][0]['timestamp'],
            reverse: bool = False) -> 'TransactionFilter':
        """결과 정렬"""
        self._filtered_data = sorted(self._filtered_data, key=key_func, reverse=reverse)
        return self

    def count(self) -> int:
        """현재 필터링된 결과의 개수 반환"""
        return len(self._filtered_data)

    def get_result(self) -> List[Dict]:
        """필터링된 결과 반환"""
        return self._filtered_data

    def get_first(self) -> Optional[Dict]:
        """첫 번째 결과 반환"""
        return self._filtered_data[0] if self._filtered_data else None
