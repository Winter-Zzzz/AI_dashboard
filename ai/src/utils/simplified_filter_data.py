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
        self._filtered_data = self._split_transactions(transactions)

    def _split_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """각 pk의 트랜잭션들을 개별 항목으로 분리"""
        result = []
        for transaction in transactions:
            for txn in transaction['transactions']:
                result.append({
                    'pk': transaction['pk'],
                    'transactions': [txn]
                })
        return result

    def reset(self) -> 'TransactionFilter':
        """필터링을 초기 상태로 리셋"""
        self._filtered_data = self._split_transactions(self.transactions)
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
                if transaction['transactions'][0]['src_pk'] == src_pk_value
            ]
        return self
    
    def by_func_name(self, func_name: int | str = -1) -> 'TransactionFilter':
        if func_name != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                if transaction['transactions'][0]['func_name'] == func_name
            ]
        return self

    def by_timestamp(self, timestamp_value: int | str = -1, compare_func: Callable = lambda x, y: x > y) -> 'TransactionFilter':
        if timestamp_value != -1:
            timestamp_datetime = datetime.fromtimestamp(int(timestamp_value))
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                if compare_func(
                    datetime.fromtimestamp(int(transaction['transactions'][0]['timestamp'])),
                    timestamp_datetime
                )
            ]
        return self

    def by_order(self, order: int = 0) -> 'TransactionFilter':
        """
        결과 정렬
        Args:
            order: 정렬 순서 (1: 내림차순, 그 외: 오름차순)
        Returns:
            TransactionFilter 인스턴스
        """
        try:
            reverse = bool(order == 1)
            self._filtered_data = sorted(
                self._filtered_data, 
                key=lambda x: x['transactions'][0]['timestamp'],
                reverse=reverse
            )
            return self
        except Exception as e:
            print(f"정렬 중 에러 발생: {str(e)}")
            return self

    def get_result(self, count: int = -1) -> List[Dict]:
        """
        필터링된 결과 반환
        Args:
            count: 반환할 트랜잭션의 개수. -1이면 전체 결과 반환
        Returns:
            필터링된 트랜잭션 리스트
        """
        if count == -1:
            return self._filtered_data
            
        return self._filtered_data[:count]
    
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


# 테스트 데이터 로드
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
json_path = os.path.join(PROJECT_ROOT, 'test', 'transaction_test.json')
data = load_json_data(json_path)

filter = TransactionFilter(data).reset().by_pk(-1).by_src_pk(-1).by_func_name(-1).by_timestamp(-1).by_order(1).get_result(7)
print(filter)