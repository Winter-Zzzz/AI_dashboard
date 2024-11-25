#!/bin/bash

# 이 스크립트를 실행하려면 먼저 실행 권한을 부여해야 합니다:
# chmod +x setup.sh
# ./setup.sh

pip install --upgrade pip

# 3. requirements.txt를 사용하여 패키지 설치
pip install -r requirements.txt

# 4. train.py 실행
echo "Training the model..."
python src/train.py

# 5. inference.py 실행
echo "Running inference..."
python src/inference.py



