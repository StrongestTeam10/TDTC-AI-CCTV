"""
server/s3_uploader.py - AWS S3 파일 업로드 및 Presigned/Public URL 관리 모듈
"""

import os
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    boto3_available = True
except ImportError:
    boto3 = None
    NoCredentialsError = Exception
    boto3_available = False

from server.config import (
    AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME
)


def get_s3_client():
    """AWS 자격 증명이 있는 경우 s3 클라이언트를 반환하고, 없으면 None 반환"""
    if not boto3_available or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        return None
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        print(f"[S3 Warning] S3 클라이언트 초기화 실패: {e}")
        return None


def upload_file_to_s3(local_file_path: str, s3_key: str = None) -> str:
    """
    로컬 파일을 AWS S3에 업로드하고 정적 Public URL을 반환합니다.
    AWS 설정이 없거나 실패하면 가상 시뮬레이션 URL을 반환하여 파이프라인 중단을 방지합니다.
    """
    if not os.path.exists(local_file_path):
        print(f"[S3 Warning] 파일이 존재하지 않습니다: {local_file_path}")
        return None

    if s3_key is None:
        s3_key = os.path.basename(local_file_path)

    # 슬래시 정규화
    s3_key = s3_key.replace("\\", "/")

    s3_client = get_s3_client()
    bucket = S3_BUCKET_NAME or "tdtc-cctv-upload"

    if s3_client is None:
        # S3 미세팅 시 시뮬레이션 모드 작동
        dummy_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"[S3 시뮬레이션] '{local_file_path}' -> {dummy_url}")
        return dummy_url

    try:
        s3_client.upload_file(
            local_file_path,
            bucket,
            s3_key
        )
        url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"[S3 업로드 성공] {url}")
        return url
    except NoCredentialsError:
        print("[S3 Warning] AWS 자격 증명을 찾을 수 없습니다. 시뮬레이션 URL을 반환합니다.")
        return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"[S3 Error] S3 업로드 중 에러: {e}")
        return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


def download_file_from_s3(s3_key: str, local_target_path: str) -> bool:
    """
    AWS S3 버킷에서 파일을 다운로드하여 로컬 경로에 저장합니다.
    """
    s3_key = s3_key.replace("\\", "/")
    s3_client = get_s3_client()
    bucket = S3_BUCKET_NAME or "tdtc-cctv-upload"

    if s3_client is None:
        print(f"[S3 Warning] S3 클라이언트를 사용할 수 없어 다운로드를 건너뜁니다: {s3_key}")
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(local_target_path)), exist_ok=True)
        print(f"[S3 Download] s3://{bucket}/{s3_key} -> '{local_target_path}' 다운로드 시작...")
        s3_client.download_file(bucket, s3_key, local_target_path)
        print(f"[S3 Download 완료] '{local_target_path}' 저장 완료 ({os.path.getsize(local_target_path) / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"[S3 Download Error] s3://{bucket}/{s3_key} 다운로드 실패: {e}")
        return False

