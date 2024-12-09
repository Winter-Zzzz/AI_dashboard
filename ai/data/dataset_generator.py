import random
import json
import os
import re

class TransactionFilterDatasetGenerator:
    def __init__(self):
        self.commands = ['Fetch', 'Get', 'Query', 'Load', 'Read', 'Pull', 'Show', 'List']
        self.sort_orders = ['recent', 'earliest']
        self.functions = ['setup function', 'on function', 'off function']
    
    def random_pk(self):
        return ''.join(random.choices('abcdef0123456789', k=130))

    def random_timestamp(self):
        return str(random.randint(1600000000, 1700000000))
    
    def generate_input(self):
        command = random.choice(self.commands)
        count = None
        sort_order = None

        conditions = []

        if random.choice([True, False]):  
            address = self.random_pk()
            conditions.append(f"to {address}")

        if random.choice([True, False]):  
            address = self.random_pk()
            conditions.append(f"from {address}")

        if random.choice([True, False]): 
            func = random.choice(self.functions)
            conditions.append(f"{func}")

        if random.choice([True, False]):  
            timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
            conditions.append(timestamp)

        if random.choice([True, False]):
            count = random.randint(1, 10)

        if random.choice([True, False]):
            sort_order = random.choice(self.sort_orders)

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
                conditions.append(f"{func}")
            elif fallback_filter == 'timestamp':
                timestamp = random.choice([f"after {self.random_timestamp()}", f"before {self.random_timestamp()}"])
                conditions.append(timestamp)

        condition = " ".join(conditions).strip()
            
        return f"{command} {count if count else ''} {sort_order if sort_order else ''} {condition}".strip()
    
    def generate_output(self, input_text):
        filter_chain = ""
        
        if "to " in input_text:
            pk = input_text.split("to ")[-1].split()[0]
            filter_chain += f".by_pk('{pk}')"
        if "from " in input_text:
            src_pk = input_text.split("from ")[-1].split()[0]
            filter_chain += f".by_src_pk('{src_pk}')"
        if any(func in input_text for func in self.functions):
            func_name = next(func for func in self.functions if func in input_text)
            func_name = func_name.replace(' function', '')
            filter_chain += f".by_func_name('{func_name}')"
        if "after " in input_text or "before " in input_text:
            timestamp = input_text.split("after ")[-1].split()[0] if "after " in input_text else input_text.split("before ")[-1].split()[0]
            filter_chain += f".by_timestamp('{timestamp}')"
        if "earliest" in input_text:
            filter_chain += ".sort()"
        elif "recent" in input_text:
            filter_chain += ".sort(reverse=True)"
        
        if re.findall(r'\b(?![0-9a-fA-F]{130}\b)(?!\d{10}\b)\d+\b', input_text):
            count = int(re.findall(r'\b(?![0-9a-fA-F]{130}\b)(?!\d{10}\b)\d+\b', input_text)[0])
        else:
            filter_chain += ".get_result()"
        
        return f"print(TransactionFilter(data){filter_chain}.get_result(){count}"
    
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


generator = TransactionFilterDatasetGenerator()
dataset = generator.generate_dataset(5000)
generated_dataset = {
    "dataset": dataset
}


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(project_root,'ai', 'data', 'raw', 'generated_dataset.json')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as json_file:
    json.dump(generated_dataset, json_file, indent=4)

print(f"JSON 파일이 {file_path}에 저장되었습니다.")