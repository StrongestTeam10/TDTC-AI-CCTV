import os
import requests
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 백엔드 알림 API 경로 및 비밀키 설정
url = "http://localhost:8080/api/ai/alerts/trigger"
api_key = os.getenv("BACKEND_API_KEY", "tdtc-super-secret-key-2026")

headers = {
    "Content-Type": "application/json",
    "X-API-KEY": api_key
}

# 테스트용 임의 알림 데이터
payload = {
    "zoneId": 1,
    "alertType": "CROWD_CRITICAL"
}

print("=" * 60)
print(" [INFO] [백엔드 알림 API(trigger) 연동 테스트 스크립트] ")
print("=" * 60)
print(f"-> Target URL: {url}")
print(f"-> Use API-KEY: {api_key}")
print(f"-> Payload: {json.dumps(payload)}")
print("-" * 60)

try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"HTTP STATUS CODE: {response.status_code}")
    print(f"RESPONSE BODY    : {response.text}")
    print("-" * 60)
    
    if response.status_code == 200:
        print("[SUCCESS] 알림 API 연동에 성공했습니다! 백엔드가 신호를 수락했습니다.")
    elif response.status_code == 401:
        print("[FAILURE] 인증 실패! X-API-KEY(백엔드 application.yml의 ai.secret-key) 값을 대조해보세요.")
    elif response.status_code == 404:
        print("[FAILURE] 경로 오류! URL 주소나 백엔드 컨트롤러 맵핑 경로를 확인하세요.")
    else:
        print("[WARNING] 백엔드에서 200 외의 예외 응답을 보냈습니다. 백엔드 콘솔 창의 에러 로그를 확인하세요.")

except requests.exceptions.ConnectionError:
    print("[CONNECTION ERROR] 백엔드 로컬 서버(http://localhost:8080)가 켜져 있지 않습니다.")
    print("   자바 백엔드 프로젝트(Spring Boot)를 구동한 뒤 다시 실행해 주세요!")
except Exception as e:
    print(f"[ERROR] 기타 에러 발생: {e}")
print("=" * 60)
