#!/bin/bash

# 이 스크립트를 실행하려면 먼저 실행 권한을 부여해야 합니다:
# chmod +x setup.sh
# ./setup.sh

# 1. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. pip 업그레이드
pip install --upgrade pip

# 3. 필수 의존성 먼저 설치
pip install numpy pandas torch transformers

# 4. requirements.txt 설치 (-r)로 한번에 설치
pip install -r requirements.txt

# 만약 오류가 발생하면 --ignore-errors 옵션 사용
pip install -r requirements.txt --ignore-errors