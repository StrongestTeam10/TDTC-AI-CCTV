# s3_uploader.py
import os
import sys
import zipfile
import boto3
from botocore.exceptions import NoCredentialsError

# .env 파일을 파싱하여 환경 변수로 주입하는 간단한 로더
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

load_env()

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")

def get_s3_client():
    """AWS 자격 증명이 있는 경우 s3 클라이언트를 반환하고, 없으면 None 반환"""
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        return None
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        print(f"[ERROR] S3 클라이언트 초기화 실패: {e}")
        return None

def upload_file_to_s3(local_file_path, s3_key=None):
    """
    로컬 파일을 AWS S3에 업로드하고 정적 Public URL을 반환합니다.
    AWS 설정이 없거나 실패하면 가상 시뮬레이션 URL을 반환하여 파이프라인 중단을 방지합니다.
    """
    if not os.path.exists(local_file_path):
        print(f"[WARNING] 파일이 존재하지 않습니다: {local_file_path}")
        return None

    if s3_key is None:
        s3_key = os.path.basename(local_file_path)

    s3_client = get_s3_client()
    if s3_client is None or not S3_BUCKET_NAME:
        # S3 미세팅 시 시뮬레이션 모드 작동
        dummy_url = f"https://mock-bucket.s3.amazonaws.com/{s3_key}"
        print(f"[INFO] [S3 시뮬레이션 모드] '{local_file_path}' -> {dummy_url}")
        return dummy_url

    try:
        # S3 업로드 실행 (Public Read 권한 부여)
        s3_client.upload_file(
            local_file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ACL": "public-read"}
        )
        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"[SUCCESS] [S3 업로드 완료] {url}")
        return url
    except NoCredentialsError:
        print("[ERROR] S3 자격 증명을 찾을 수 없습니다. 시뮬레이션 URL을 반환합니다.")
        return f"https://mock-bucket.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"[ERROR] S3 업로드 에러: {e}")
        return f"https://mock-bucket.s3.amazonaws.com/{s3_key}"

def archive_local_csv_to_zip(csv_paths, zip_name="results_archive.zip"):
    """결과 CSV 파일들을 로컬에서 Zip으로 아카이빙하고 S3로 업로드합니다."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "results")
    zip_path = os.path.join(results_dir, zip_name)

    # Zip 압축 생성
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                zipf.write(csv_path, os.path.basename(csv_path))
                print(f"[INFO] 아카이브 추가: {os.path.basename(csv_path)}")

    print(f"[SUCCESS] 압축 완료: {zip_path}")
    
    # S3 백업 전송
    s3_url = upload_file_to_s3(zip_path, f"backups/{zip_name}")
    
    return s3_url

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("[TEST] [S3 Uploader 테스트 모드 구동]")
        print(f"[INFO] Loaded Bucket: {S3_BUCKET_NAME}")
        
        # 임시 텍스트 파일 업로드 테스트
        test_file = "s3_test_temp.txt"
        with open(test_file, "w") as f:
            f.write("S3 Uploader Test Content")
            
        print("1. 단일 파일 업로드 테스트...")
        url = upload_file_to_s3(test_file, "tests/s3_test_temp.txt")
        print(f"결과 URL: {url}")
        
        if os.path.exists(test_file):
            os.remove(test_file)
            
        print("\n2. Zip 아카이빙 테스트...")
        dummy_csv = "dummy_data.csv"
        with open(dummy_csv, "w") as f:
            f.write("id,value\n1,test")
        
        archive_url = archive_local_csv_to_zip([dummy_csv], "test_archive.zip")
        print(f"아카이브 URL: {archive_url}")
        
        if os.path.exists(dummy_csv):
            os.remove(dummy_csv)
        archive_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"), "test_archive.zip")
        if os.path.exists(archive_path):
            os.remove(archive_path)
