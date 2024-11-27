import random
import json
import os

class TransactionFilterDatasetGenerator:
    def __init__(self):
        self.functions = ['setup', 'on', 'off']
        self.commands = ['Show', 'Find', 'Get', 'Display']
        self.transaction_types = ['transactions', 'function calls']
        self.sort_orders = ['earliest', 'latest', 'most recent', 'oldest']
        self.keys = ['pk', 'src_pk', 'func_name']
    
    def random_pk(self):
        return ''.join(random.choices('abcdef0123456789', k=130))

    def random_timestamp(self):
        return str(random.randint(1600000000, 1700000000))
    
    def generate_input(self):
        command = random.choice(self.commands)
        count = random.randint(1, 10)
        tx_type = random.choice(self.transaction_types)
        sort_order = random.choice(self.sort_orders)
        
        # 필터 랜덤 적용
        conditions = []
        if random.choice([True, False]):  # pk 조건 랜덤
            address = self.random_pk()
            conditions.append(f"to {address}" if tx_type == 'transactions' else f"to {address}")
        if random.choice([True, False]):  # src_pk 조건 랜덤
            address = self.random_pk()
            conditions.append(f"from {address}" if tx_type == 'transactions' else f"from {address}")
        if random.choice([True, False]):  # func_name 조건 랜덤
            func = random.choice(self.functions)
            conditions.append(f"{func} {tx_type}")
        if random.choice([True, False]):  # timestamp 조건 랜덤
            timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
            conditions.append(timestamp)
        
        # 최소 하나의 필터 강제 포함
        if not conditions:
            fallback_filter = random.choice(['to', 'from', 'func', 'timestamp'])
            if fallback_filter == 'to':
                address = self.random_pk()
                conditions.append(f"to {address}")
            elif fallback_filter == 'from':
                address = self.random_pk()
                conditions.append(f"from {address}")
            elif fallback_filter == 'func':
                func = random.choice(self.functions)
                conditions.append(f"{func} {tx_type}")
            elif fallback_filter == 'timestamp':
                timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
                conditions.append(timestamp)

        condition = " ".join(conditions).strip()
        return f"{command} {count} {sort_order} {condition}".strip()
    
    def generate_output(self, input_text):
        filter_chain = "TransactionFilter(data)"
        
        if "to " in input_text:
            pk = input_text.split("to ")[-1].split()[0]
            filter_chain += f".by_pk('{pk}')"
        if "from " in input_text:
            src_pk = input_text.split("from ")[-1].split()[0]
            filter_chain += f".by_src_pk('{src_pk}')"
        if any(func in input_text for func in self.functions):
            func_name = next(func for func in self.functions if func in input_text)
            filter_chain += f".by_func_name('{func_name}')"
        if "after " in input_text or "before " in input_text:
            timestamp = input_text.split("after ")[-1].split()[0] if "after " in input_text else input_text.split("before ")[-1].split()[0]
            filter_chain += f".by_timestamp('{timestamp}')"
        if "earliest" in input_text or "oldest" in input_text:
            filter_chain += ".sort()"
        elif "latest" in input_text or "most recent" in input_text:
            filter_chain += ".sort(reverse=True)"
        
        if random.choice([True, False]):  # count 슬라이싱 랜덤 적용
            count = int(input_text.split()[1])
            filter_chain += f".get_result()[:{count}]"
        else:
            filter_chain += ".get_result()"
        
        return f"print({filter_chain})"
    
    def generate_dataset(self, n=10):
        dataset = []
        for _ in range(n):
            input_text = self.generate_input()
            output_text = self.generate_output(input_text)
            dataset.append({
                "input": input_text,
                "output": output_text
            })
        return dataset

# 예시 사용
generator = TransactionFilterDatasetGenerator()
dataset = generator.generate_dataset(50)
generated_dataset = {
    "datasets": dataset
}

# for entry in dataset:
#     print(entry)

file_path = './ai/data/raw/generated_dataset.json'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"JSON 파일이 {file_path}에 저장되었습니다.")