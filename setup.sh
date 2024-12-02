#!/bin/bash

# 이 스크립트를 실행하려면 먼저 실행 권한을 부여해야 합니다:
# chmod +x setup.sh
# ./setup.sh

python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# 3. requirements.txt를 사용하여 패키지 설치
pip install -r requirements.txt

# 4. FastAPI.py 실행 (서버시작)
python3 ai/src/FastAPI.py