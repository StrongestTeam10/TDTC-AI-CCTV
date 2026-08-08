import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# DB 접속 정보 파싱
# jdbc:postgresql://aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres?sslmode=require 형태를 파싱하거나 기본값 사용
DB_HOST = os.environ.get("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres.uusnthiedfwzlcsgtnhn")

# 2026-08-08: 공개 저장소라 비밀번호 하드코딩을 제거했다.
# 반드시 루트 .env 또는 환경변수로 주입할 것.
DB_PASS = os.environ.get("DB_PASSWORD", "")
if not DB_PASS:
    print("[WARNING] DB_PASSWORD 환경변수가 비어 있습니다. 루트 .env 파일을 확인하세요.")

print("=" * 70)
print(" [INFO] [Supabase DB - 긴급 신고 & S3 연동 적재 데이터 간편 조회] ")
print("=" * 70)
print(f"-> DB Host: {DB_HOST}")
print(f"-> DB User: {DB_USER}")
print("-" * 70)

query = """
SELECT 
    a.alert_id, 
    a.zone_id, 
    a.alert_type, 
    a.alerted_at,
    r.s3_pdf_url, 
    v.s3_clip_url
FROM emgalrt01h a
LEFT JOIN pstrprt01h r ON a.alert_id = r.alert_id
LEFT JOIN vdoclip01m v ON r.video_id = v.clip_id
ORDER BY a.alerted_at DESC 
LIMIT 5;
"""

try:
    # DB 연결
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        sslmode="require"
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("[INFO] DB에 적재된 긴급 신고 이력(emgalrt01h)이 하나도 없습니다.")
        print("   먼저 test_alert_api.py를 실행하여 백엔드로 알림을 쏴 주세요!")
    else:
        for idx, row in enumerate(rows, 1):
            print(f" [{idx}] 신고 ID: {row['alert_id']} | 구역: {row['zone_id']} | 종류: {row['alert_type']}")
            print(f"     * 신고 시각 : {row['alerted_at']}")
            print(f"     * PDF 분석서: {row['s3_pdf_url'] if row['s3_pdf_url'] else '[없음]'}")
            print(f"     * 증거 영상 : {row['s3_clip_url'] if row['s3_clip_url'] else '[없음]'}")
            print("-" * 70)

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] 데이터베이스 접속 오류: {e}")
    print("   .env 파일의 DB 설정값이나 네트워크 상태를 확인하세요.")
print("=" * 70)
