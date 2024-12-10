#!/bin/bash

# 1. 가상환경 생성
echo "1. 가상환경 생성 중..."
python3.11 -m venv venv
source venv/bin/activate

# 2. pip 업그레이드
echo "2. pip 업그레이드 중..."
pip install --upgrade pip

# 3. requirements.txt 설치
echo "3. requirements.txt 설치 중..."
pip install -r requirements.txt

# 4. FastAPI 서버 실행 준비
echo "4. FastAPI 서버 실행 준비 중..."
PORT=8000

# 8000번 포트 사용 중인지 확인
if lsof -i :$PORT; then
    echo "포트 $PORT가 이미 사용 중입니다. 프로세스를 종료합니다."
    PID=$(lsof -t -i :$PORT)
    kill -9 $PID
    echo "포트 $PORT의 프로세스($PID)를 종료했습니다."
fi

# FastAPI 서버 실행
echo "FastAPI 서버 실행 중..."
cd ai/src
python FastAPI.py > fastapi.log 2>&1 &
FASTAPI_PID=$!  # FastAPI 서버 프로세스 ID 저장
echo "FastAPI 서버 실행 중... 로그는 fastapi.log 파일에 저장됩니다."

# 5. Frontend 디렉터리로 이동 및 설치
cd ../../Frontend
npm install
npm start &
FRONTEND_PID=$!  # npm 서버 프로세스 ID 저장

# 종료 명령 대기
echo "모든 서비스가 실행되었습니다. 종료하려면 Ctrl+C를 누르십시오."
trap "echo '종료 중...'; kill $FASTAPI_PID $FRONTEND_PID; deactivate; exit 0" SIGINT

wait


# FastAPI 로그 확인
# tail -f ai/src/fastapi.log