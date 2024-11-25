#!/bin/bash

# 이 스크립트를 실행하려면 먼저 실행 권한을 부여해야 합니다:
# chmod +x setup.sh
# ./setup.sh

# 1. 가상 환경 만들기
python3.11 -m venv venv

# 2. 가상 환경 활성화
source venv/bin/activate  # macOS/Linux 경우

pip install --upgrade pip

# 3. requirements.txt를 사용하여 패키지 설치
pip install -r requirements.txt

# 4. train.py 실행
echo "Training the model..."
python src/train.py

# 5. inference.py 실행
echo "Running inference..."
python src/inference.py
