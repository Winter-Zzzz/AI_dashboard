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
        """필터링을 초기 상태로 리셋"""
        self._filtered_data = self.transactions
        return self

    def by_pk(self, pk_value: int | str = -1) -> 'TransactionFilter':
        """pk로 필터링"""
        if pk_value != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data 
                if transaction['pk'] == pk_value
            ]
        return self

    def by_src_pk(self, src_pk_value: int | str = -1) -> 'TransactionFilter':
        """src_pk로 필터링"""
        if src_pk_value != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                for txn in transaction['transactions']
                if txn['src_pk'] == src_pk_value
            ]
        return self
    
    def by_func_name(self, func_name: int | str = -1) -> 'TransactionFilter':
        """func_name으로 필터링"""
        if func_name != -1:
            self._filtered_data = [
                transaction for transaction in self._filtered_data
                for txn in transaction['transactions']
                if txn['func_name'] == func_name
            ]
        return self

    def by_timestamp(self, timestamp_value: int | str = -1, compare_func: Callable = lambda x, y: x > y) -> 'TransactionFilter':
        """timestamp로 필터링. 비교 함수를 커스텀할 수 있음"""
        if timestamp_value != -1:
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

    def sort(self, 
            key_func: Optional[Callable[[Dict], Any]] = None,
            order: int = 0) -> 'TransactionFilter':
        """
        결과 정렬
        Args:
            key_func: 정렬 키 함수. None이면 기본 정렬 함수 사용
            order: 정렬 순서 (0: 오름차순, 1: 내림차순)
        Returns:
            TransactionFilter 인스턴스 (메서드 체이닝을 위해)
        """
        try:
            # 기본 정렬 함수 정의
            def default_key(x: Dict) -> Any:
                return x['transactions'][0]['timestamp']
            
            # key_func가 숫자면 order로 처리하고 default_key 사용
            if isinstance(key_func, (int, float)):
                order = key_func
                actual_key = default_key
            else:
                actual_key = default_key if key_func in [-1, None] else key_func
            
            # order 값에 따라 reverse 설정
            reverse = bool(order == 1)
            
            self._filtered_data = sorted(self._filtered_data, key=actual_key, reverse=reverse)
            return self
        except Exception as e:
            print(f"정렬 중 에러 발생: {str(e)}")
            return self
    
    def get_result(self, count: int = -1) -> List[Dict]:
        """필터링된 결과 반환"""
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


# 테스트 데이터 로드
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
json_path = os.path.join(PROJECT_ROOT, 'test', 'transaction_test.json')
data = load_json_data(json_path)

filter = TransactionFilter(data).reset().by_pk('04fcc1da3dc60d7e4e4c987022c9f08f20b2e9b16df6cd6bcb9b1251021b018260d603df7809991b9e21ee0263f92dad29fd2f1de56851ab62b2ecb410db0389a0').by_src_pk(-1).by_func_name(-1).by_timestamp(-1).sort(1).get_result(-1)
print(filter)