import random
import json
import os
import re

class TransactionFilterDatasetGenerator:
    def __init__(self):
        self.commands = ['Fetch', 'Get', 'Query', 'Load', 'Read', 'Pull', 'Show', 'List']
        self.orders = ['latest', 'oldest', 'recent', 'earliest', 'most recent']
        self.functions = [
            'feedback function', 
            'setup function', 
            'getTemperature function', 
            'setTemperatureMode function',
            'getDeviceStatus function',
            'getPowerState function',
            'setPowerState function',
            'resetDevice function',
            'getNetworkInfo function',
            'getWiFiStatus function',
            'reconnectNetwork function',
            'updateFirmware function',
            'getHumidity function',
            'getPressure function',
            'getBatteryLevel function',
            'getLastMeasurement function',
            'getSensorHistory function',
            'getSettings function',
            'setMeasurementUnit function',
            'setMeasureInterval function',
            'setAlarmThreshold function',
            'setAnimal function',
            'getCoordinate function',
        ]
        self.transaction_words = ['', 'transaction', 'transactions', 'txn', 'txns']
        self.number_words = {
            1: ['one', '1'],
            2: ['two', '2'],
            3: ['three', '3'],
            4: ['four', '4'],
            5: ['five', '5'],
            6: ['six', '6'],
            7: ['seven', '7'],
            8: ['eight', '8'],
            9: ['nine', '9'],
            10: ['ten', '10']
        }
    
    def get_random_count(self):
        """숫자나 단어로 된 카운트 반환"""
        if random.choice([True, False]):  # 숫자를 사용할지 단어를 사용할지 결정
            number = random.randint(1, 10)
            # None, number word 또는 숫자 문자열 중 하나 랜덤 선택
            choices = [None, random.choice(self.number_words[number])]
            return random.choice(choices)
        return 'all'
    
    def word_to_number(self, word):
        """숫자 단어를 숫자로 변환"""
        if word is None or word == 'all':
            return word
        
        # 이미 숫자 문자열이면 그대로 반환
        if str(word).isdigit():
            return word
            
        # 숫자 단어를 숫자로 변환
        for num, words in self.number_words.items():
            if word.lower() in [w.lower() for w in words]:
                return str(num)
        return word
    
    def random_pk(self):
        return ''.join(random.choices('abcdef0123456789', k=130))

    def random_timestamp(self):
        return str(random.randint(1730780906, 1800000000))
    
    def generate_input(self):
        command = random.choice(self.commands)
        transaction_word = random.choice(self.transaction_words)
        count = self.get_random_count() 
        order = random.choice(self.orders) if random.choice([True, False]) else None
        # order = random.choice(self.orders) # generate_balanced_dataset
        
        # 조건들 기본적으로 설정
        conditions = []

        # 조건이 없을 경우 None 값으로 대체하고 조건을 생성
        to_address = self.random_pk() if random.choice([True, False]) else None
        from_or_by_address = self.random_pk() if random.choice([True, False]) else None
        func = random.choice(self.functions) if random.choice([True, False]) else None
        timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"]) if random.choice([True, False]) else None
        
        # 조건이 None이 아니면 추가
        if to_address:
            conditions.append(f"to {to_address}")
        if from_or_by_address:
            if random.choice([True, False]):
                conditions.append(f"from {from_or_by_address}")
            else:
                conditions.append(f"by {from_or_by_address}")
        if func:
            conditions.append(f"{func}")

        if random.choice([True, False]):
            if random.choice([True, False]):  # between vs single timestamp
                time1 = self.random_timestamp()
                time2 = self.random_timestamp()
                start_time = min(int(time1), int(time2))
                end_time = max(int(time1), int(time2))
                conditions.append(f"between {start_time} and {end_time}")
            else:
                timestamp = self.random_timestamp()
                conditions.append(random.choice([f"after {timestamp}", f"before {timestamp}"]))
        
        # 조건이 하나도 없으면, 하나 이상을 강제로 추가
        if not conditions:
            fallback_filter = random.choice(['to', 'from', 'by', 'func', 'timestamp'])
            if fallback_filter == 'to':
                address = self.random_pk()
                conditions.append(f"to {address}")
            elif fallback_filter == 'from' or 'by':
                address = self.random_pk()
                if random.choice([True, False]):
                    conditions.append(f"from {address}")
                else:
                    conditions.append(f"by {address}")
            elif fallback_filter == 'func':
                func = random.choice(self.functions)
                conditions.append(f"{func}")
            elif fallback_filter == 'timestamp':
                timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
                conditions.append(timestamp)

        # 조건들을 조합하여 반환
        condition = " ".join(conditions).strip()

        parts = [command]
        if count is not None:
            parts.append(str(count))
        # order와 transaction_word를 함께 처리
        if order and transaction_word:
            parts.append(f"{order} {transaction_word}")
        elif order:
            parts.append(order)
        elif transaction_word:
            parts.append(transaction_word)
        parts.append(condition)

        return " ".join(parts).strip(), count
    
    def generate_output(self, input_text):
        filter_chain = ""
        
        # 입력 텍스트에서 pk 값 추출
        if "to " in input_text:
            pk = input_text.split("to ")[-1].split()[0]
            filter_chain += f".by_pk('{pk}')"
        else:
            filter_chain += ".by_pk(-1)"
        
        if "from " in input_text:
            src_pk = input_text.split("from ")[-1].split()[0]
            filter_chain += f".by_src_pk('{src_pk}')"
        elif "by " in input_text:
            src_pk = input_text.split("by ")[-1].split()[0]
            filter_chain += f".by_src_pk('{src_pk}')"
        else:
            filter_chain += ".by_src_pk(-1)"
        
        if any(func in input_text for func in self.functions):
            func_name = next(func for func in self.functions if func in input_text)
            func_name = func_name.replace(' function', '')
            filter_chain += f".by_func_name('{func_name}')"
        else:
            filter_chain += ".by_func_name(-1)"
        
        # timestamp 처리 부분 수정
        if "between " in input_text:
            match = re.search(r"between (\d+) and (\d+)", input_text)
            if match:
                start_time, end_time = match.groups()
                filter_chain += f".between('{start_time}', '{end_time}')"
        elif "after " in input_text:
            timestamp = input_text.split("after ")[-1].split()[0]
            filter_chain += f".after('{timestamp}')"
        elif "before " in input_text:
            timestamp = input_text.split("before ")[-1].split()[0]
            filter_chain += f".before('{timestamp}')"
        

        input_lower = input_text.lower()
        if "oldest" in input_lower or "earliest" in input_lower:
            filter_chain += ".by_order(0)"
        elif "most recent" in input_lower:
            filter_chain += ".by_order(1)"
        elif "latest" in input_lower or "recent" in input_lower:
            filter_chain += ".by_order(1)"
        else:
            filter_chain += ".by_order(0)"
        
        return filter_chain
    
    def generate_dataset(self, n=10):
        dataset = []
        for _ in range(n):
            input_text, count = self.generate_input()
            input_lower = input_text.lower()
            
            # plural과 order keyword 확인
            has_plural = 'txns' in input_lower or 'transactions' in input_lower
            has_order = any(order in input_lower for order in self.orders)
            
            # count 결정 - 숫자로 변환
            if count == 'all' or (has_plural and count is None):
                count_value = -1
            elif count is None and has_order:
                count_value = 1
            elif count is None:
                count_value = -1
            else:
                # 숫자 단어나 문자열을 실제 숫자로 변환
                count_value = self.word_to_number(count)
                
            output_text = f"txn{self.generate_output(input_text)}.get_result({count_value})"
            dataset.append({
                "input": input_text,
                "output": output_text
            })
        return {"dataset": dataset}
    
    def generate_balanced_dataset(self, n=5000):
        """
        처음부터 order(0)과 order(1)이 1:1 비율로 균형잡힌 데이터셋 생성
        
        Args:
            n (int): 생성할 전체 데이터 수 (각 order 타입별로 n/2개씩 생성됨)
            
        Returns:
            dict: 균형잡힌 데이터셋
        """
        dataset = []
        orders_per_type = n // 2

        # order(0) 데이터 생성
        for _ in range(orders_per_type):
            command = random.choice(self.commands)
            transaction_word = random.choice(self.transaction_words)
            count = self.get_random_count()
            order = random.choice(["oldest", "earliest"])  # order(0)용 키워드
            
            input_text, count = self.generate_input()
            # order 키워드를 강제로 바꾸기
            input_text = re.sub(r'\b(latest|recent|earliest|oldest)\b', order, input_text, flags=re.IGNORECASE)
            if not any(word in input_text.lower() for word in ["oldest", "earliest"]):
                input_text = f"{order} {input_text}"
                
            # output 생성 및 추가
            input_lower = input_text.lower()
            has_plural = 'txns' in input_lower or 'transactions' in input_lower
            if count == 'all' or (has_plural and count is None):
                count_value = "-1"
            elif count is None:
                count_value = "1"
            else:
                count_value = self.word_to_number(count)
                
            output_text = f"txn{self.generate_output(input_text)}.get_result({count_value})"
            dataset.append({"input": input_text, "output": output_text})

        # order(1) 데이터 생성
        for _ in range(orders_per_type):
            command = random.choice(self.commands)
            transaction_word = random.choice(self.transaction_words)
            count = self.get_random_count()
            order = random.choice(["latest", "most recent", "recent"])  # order(1)용 키워드
            
            input_text, count = self.generate_input()
            # order 키워드를 강제로 바꾸기
            input_text = re.sub(r'\b(latest|recent|earliest|oldest)\b', order, input_text, flags=re.IGNORECASE)
            if not any(word in input_text.lower() for word in ["latest", "most recent", "recent"]):
                input_text = f"{order} {input_text}"
                
            # output 생성 및 추가
            input_lower = input_text.lower()
            has_plural = 'txns' in input_lower or 'transactions' in input_lower
            if count == 'all' or (has_plural and count is None):
                count_value = "-1"
            elif count is None:
                count_value = "1"
            else:
                count_value = self.word_to_number(count)
                
            output_text = f"txn{self.generate_output(input_text)}.get_result({count_value})"
            dataset.append({"input": input_text, "output": output_text})

        # 데이터셋 섞기
        random.shuffle(dataset)
        return {"dataset": dataset}


# Dataset 생성
generator = TransactionFilterDatasetGenerator()
dataset = generator.generate_dataset(10000)
# dataset = generator.generate_balanced_dataset(5000) 

# 파일 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(project_root, 'ai', 'data', 'raw', 'simplified_generated_dataset.json')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# JSON 파일로 저장
with open(file_path, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)


# 데이터셋 통계 출력
print(f"데이터셋 통계:")
print(f"- 전체 데이터 수: {len(dataset['dataset'])}\n")

# by_order 통계
order_0_count = sum(1 for item in dataset['dataset'] if 'by_order(0)' in item['output'])
order_1_count = sum(1 for item in dataset['dataset'] if 'by_order(1)' in item['output'])
print("Order 통계:")
print(f"- by_order(0) 수: {order_0_count}")
print(f"- by_order(1) 수: {order_1_count}")

# by_pk 통계
pk_valid = sum(1 for item in dataset['dataset'] if "by_pk(-1)" not in item['output'])
pk_default = sum(1 for item in dataset['dataset'] if "by_pk(-1)" in item['output'])
print("PK 통계:")
print(f"- 유효한 by_pk 수: {pk_valid}")
print(f"- 기본값(-1) 수: {pk_default}")

# by_src_pk 통계
src_pk_valid = sum(1 for item in dataset['dataset'] if "by_src_pk(-1)" not in item['output'])
src_pk_default = sum(1 for item in dataset['dataset'] if "by_src_pk(-1)" in item['output'])
print("Source PK 통계:")
print(f"- 유효한 by_src_pk 수: {src_pk_valid}")
print(f"- 기본값(-1) 수: {src_pk_default}")

# by_func_name 통계
func_valid = sum(1 for item in dataset['dataset'] if "by_func_name(-1)" not in item['output'])
func_default = sum(1 for item in dataset['dataset'] if "by_func_name(-1)" in item['output'])
print("Function 통계:")
print(f"- 유효한 by_func_name 수: {func_valid}")
print(f"- 기본값(-1) 수: {func_default}")

# Timestamp 통계
between_count = sum(1 for item in dataset['dataset'] if 'between' in item['output'])
after_count = sum(1 for item in dataset['dataset'] if 'after' in item['output'])
before_count = sum(1 for item in dataset['dataset'] if 'before' in item['output'])
no_timestamp = len(dataset['dataset']) - (between_count + after_count + before_count)
print("Timestamp 통계:")
print(f"- between 수: {between_count}")
print(f"- after 수: {after_count}")
print(f"- before 수: {before_count}")
print(f"- timestamp 없음: {no_timestamp}\n")

print(f"JSON 파일이 {file_path}에 저장되었습니다.")


# 데이터셋 통계:
# - 전체 데이터 수: 5000

# Order 통계:
# - by_order(0) 수: 3479
# - by_order(1) 수: 1521
# PK 통계:
# - 유효한 by_pk 수: 2534
# - 기본값(-1) 수: 2466
# Source PK 통계:
# - 유효한 by_src_pk 수: 2743
# - 기본값(-1) 수: 2257
# Function 통계:
# - 유효한 by_func_name 수: 2445
# - 기본값(-1) 수: 2555
# Timestamp 통계:
# - between 수: 1233
# - after 수: 618
# - before 수: 636
# - timestamp 없음: 2513

