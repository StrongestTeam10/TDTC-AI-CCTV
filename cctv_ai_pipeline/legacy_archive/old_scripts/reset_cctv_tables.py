# reset_cctv_tables.py
# CCTV / 보행자 관제 전용 DB 테이블 핀포인트 초기화 스크립트

import os
import sys
import urllib.parse

# 1. DB 접속 정보 로드 (TDTC-AI-BE/.env 참조)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
BE_ENV_PATH = os.path.join(WORKSPACE_DIR, "TDTC-AI-BE", ".env")

db_url = None
db_user = None
db_password = None

if os.path.exists(BE_ENV_PATH):
    with open(BE_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DEV_DB_URL="):
                db_url = line.split("=", 1)[1]
            elif line.startswith("DEV_DB_USERNAME="):
                db_user = line.split("=", 1)[1]
            elif line.startswith("DEV_DB_PASSWORD="):
                db_password = line.split("=", 1)[1]

# 디폴트 폴백 설정
if not db_url:
    db_url = os.environ.get("DEV_DB_URL", "jdbc:postgresql://aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres?sslmode=require")
if not db_user:
    db_user = os.environ.get("DEV_DB_USERNAME", "postgres.uusnthiedfwzlcsgtnhn")
if not db_password:
    db_password = os.environ.get("DEV_DB_PASSWORD", "tdtcStrong10!")

# jdbc:postgresql:// -> postgresql:// 변환
if db_url.startswith("jdbc:"):
    db_url = db_url[5:]

print("=" * 70)
print(" [CCTV 관제 전용 DB 테이블 핀포인트 초기화 스크립트] ")
print("=" * 70)
print(f"[INFO] DB Target Host: {db_url}")
print(f"[INFO] DB User: {db_user}")

TARGET_TABLES = [
    "pedaggr01h",
    "sf_frame_summary",
    "sf_pedestrian_history",
    "sf_roi_risk_log"
]

print("\n[초기화(TRUNCATE) 대상 테이블]")
for idx, tbl in enumerate(TARGET_TABLES, 1):
    print(f"  {idx}. {tbl}")
print("[NOTICE] User, Market, Post, Facility 등 타 서비스 테이블은 100% 보존됩니다.\n")


try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("📦 psycopg2 패키지 설치 시도 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2
    from psycopg2 import sql

def reset_tables():
    parsed = urllib.parse.urlparse(db_url)
    host = parsed.hostname
    port = parsed.port or 5432
    dbname = parsed.path.lstrip("/") or "postgres"

    print(f"[INFO] Supabase DB 연결 시도 중 ({host}:{port}/{dbname})...")
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=db_user,
        password=db_password,
        sslmode="require"
    )
    conn.autocommit = True
    cursor = conn.cursor()
    print("[SUCCESS] Supabase DB 연결 성공!")

    success_count = 0
    for tbl in TARGET_TABLES:
        try:
            cursor.execute(f"TRUNCATE TABLE {tbl} RESTART IDENTITY CASCADE;")
            print(f"  [TRUNCATE SUCCESS] {tbl}")
            success_count += 1
        except Exception as e:
            print(f"  [TRUNCATE SKIP/NOTE] {tbl}: {e}")

    cursor.close()
    conn.close()
    print("-" * 70)
    print(f"[SUCCESS] 총 {success_count}개 관제 전용 테이블 초기화 완료!")
    print("-" * 70)


if __name__ == "__main__":
    reset_tables()
