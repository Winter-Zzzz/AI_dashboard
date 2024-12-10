import random
import json
import os
import re

class TransactionFilterDatasetGenerator:
    def __init__(self):
        self.commands = ['Fetch', 'Get', 'Query', 'Load', 'Read', 'Pull', 'Show', 'List']
        self.orders = ['latest', 'oldest', 'recent', 'earliest', 'most recent', 'last']
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
        self.dataset = {"dataset": []}
    
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
        elif "latest" in input_lower or "recent" in input_lower or "last" in input_lower:
            filter_chain += ".by_order(1)"
        else:
            filter_chain += ".by_order(0)"
        
        return filter_chain
    
    def generate_dataset(self, n=10):
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
            self.dataset["dataset"].append({"input": input_text, "output": output_text})

        return len(self.dataset["dataset"])

    
    def generate_most_recent_dataset(self, n=10):
        """
        'most recent', 'first recent', 'second recent', 'third recent'에 대한
        필터링된 데이터셋을 생성하는 함수
        """
        self.transaction_words = ['txn', 'none', 'txns']
        recent_orders = ['most recent', 'first recent', 'second recent', 'third recent']
        
        for order in recent_orders:
            for _ in range(n):
                command = random.choice(self.commands)
                transaction_word = random.choice(self.transaction_words)
                # none일 경우 빈 문자열로 변경
                if transaction_word == 'none':
                    transaction_word = ''
                
                # 필터 조건 생성
                conditions = []
                
                # 필터 조건 추가 (랜덤으로 1-3개)
                num_conditions = random.randint(1, 3)
                available_filters = ['to', 'from/by', 'func', 'timestamp']
                selected_filters = random.sample(available_filters, num_conditions)
                
                for filter_type in selected_filters:
                    if filter_type == 'to':
                        conditions.append(f"to {self.random_pk()}")
                    elif filter_type == 'from/by':
                        address = self.random_pk()
                        conditions.append(f"{'from' if random.choice([True, False]) else 'by'} {address}")
                    elif filter_type == 'func':
                        func = random.choice(self.functions)
                        conditions.append(func)
                    elif filter_type == 'timestamp':
                        if random.choice([True, False]):
                            time1, time2 = self.random_timestamp(), self.random_timestamp()
                            start_time = min(int(time1), int(time2))
                            end_time = max(int(time1), int(time2))
                            conditions.append(f"between {start_time} and {end_time}")
                        else:
                            timestamp = self.random_timestamp()
                            conditions.append(random.choice([f"after {timestamp}", f"before {timestamp}"]))
                
                # 입력 텍스트 생성 (transaction_word가 비어있으면 공백 없이)
                input_parts = [command]
                input_parts.append(order)
                if transaction_word:
                    input_parts.append(transaction_word)
                input_parts.extend(conditions)
                input_text = " ".join(input_parts)
                
                # cnt 설정
                cnt = 0 if order in ['most recent', 'first recent'] else (1 if order == 'second recent' else 2)
                
                # output 생성
                output_text = f"txn{self.generate_output(input_text)}.get_result(1)[{cnt}]"
                self.dataset["dataset"].append({"input": input_text, "output": output_text})

        return len(self.dataset["dataset"])
# Dataset 생성
generator = TransactionFilterDatasetGenerator()
dataset = generator.generate_dataset(400)
total_size = generator.generate_most_recent_dataset(25)
print(total_size)


# 파일 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(project_root, 'ai', 'data', 'raw', 'simplified_generated_dataset.json')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# JSON 파일로 저장
with open(file_path, 'w') as json_file:
    json.dump(generator.dataset, json_file, indent=4)
