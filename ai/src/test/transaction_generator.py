import random
import time
import secrets
import json
import os

def generate_random_hex(length=64):
    """Generate a random hex string of specified length"""
    return secrets.token_hex(length // 2)

def generate_public_key():
    """Generate a random public key starting with '04'"""
    return "04" + generate_random_hex(128)

def generate_raw_data():
    """Generate random raw data similar to the example"""
    parts = [
        "6337b0bdd7a7f41fc0dae95d2472ee28",
        "648d371c92a49eba91aad29c8e7a0845",
        "f227a163546181e9d87a60e660c73cc9",
        "983139404f4dcd9d12380e8368b652d3",
        "666565646261636b000000000000000000",
        generate_random_hex(64),
        generate_random_hex(32)
    ]
    return "".join(parts)

def generate_transaction_data(num_pks=6, transactions_per_src=3, base_timestamp=None):
    """
    Generate transaction data where each pk has multiple src_pks,
    and each src_pk has different function names
    
    Args:
        num_pks (int): Number of different main pks
        transactions_per_src (int): Number of transactions per src_pk
        base_timestamp (int): Starting timestamp (optional)
    """
    if base_timestamp is None:
        base_timestamp = int(time.time())
    
    function_names = ["setup", "on", "off"]
    result = []
    current_timestamp = base_timestamp
    
    # 각 pk에 대해
    for _ in range(num_pks):
        main_pk = generate_public_key()
        
        # 각 pk마다 3개의 src_pk 생성
        for _ in range(3):  # 3개의 서로 다른 src_pk
            src_pk = generate_public_key()
            
            # 각 src_pk마다 3개의 트랜잭션 (setup, on, off)
            for i in range(transactions_per_src):
                transaction = {
                    "pk": main_pk,
                    "transactions": [
                        {
                            "raw_data": generate_raw_data(),
                            "src_pk": src_pk,
                            "timestamp": str(current_timestamp),
                            "func_name": function_names[i % len(function_names)]
                        }
                    ]
                }
                result.append(transaction)
                current_timestamp += 1
    
    return result

def save_to_json(data, filename='transaction_test.json'):
    """Save data to JSON file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {filepath}")

# 실행 코드
if __name__ == "__main__":
    base_timestamp = 1733045250
    transactions = generate_transaction_data(
        num_pks=6,
        transactions_per_src=3,
        base_timestamp=base_timestamp
    )
    
    # JSON 파일로 저장
    save_to_json(transactions)