"""
upload_sources_to_s3.py - 로컬 원본 영상을 AWS S3 (source-videos/)에 일괄 업로드
"""

import os
import sys
import boto3
from dotenv import load_dotenv

# TDTC-AI-CCTV .env 로드
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME", "tdtc-cctv-upload")

def get_s3_client():
    if not (AWS_ACCESS_KEY and AWS_SECRET_KEY):
        raise ValueError("AWS 자격 증명(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)이 누락되었습니다.")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        sys.stdout.write(
            f"\r-> 업로드 중: {os.path.basename(self._filename)}  {percentage:.2f}% ({int(self._seen_so_far/(1024*1024))}/{int(self._size/(1024*1024))} MB)"
        )
        sys.stdout.flush()

def upload_sources():
    s3 = get_s3_client()
    workspace_root = os.path.dirname(BASE_DIR)
    
    zone_sources = {
        1: os.path.join(workspace_root, "zone", "zone_id1", "test_south01.mp4"),
        2: os.path.join(workspace_root, "zone", "zone_id2", "test_center01.mp4"),
        3: os.path.join(workspace_root, "zone", "zone_id3", "test_north01.mp4"),
    }

    print(f"=== AWS S3 원본 비디오 일괄 업로드 시작 (버킷: {S3_BUCKET_NAME}) ===")
    
    for zone_id, local_path in zone_sources.items():
        s3_key = f"source-videos/zone{zone_id}_source.mp4"
        if not os.path.exists(local_path):
            print(f"[경고] 로컬 파일을 찾을 수 없습니다: {local_path}")
            continue

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"\n[Zone {zone_id}] {local_path} ({file_size_mb:.2f} MB) -> s3://{S3_BUCKET_NAME}/{s3_key}")
        
        progress = ProgressPercentage(local_path)
        s3.upload_file(
            Filename=local_path,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
            Callback=progress
        )
        print(f"\n  [OK] Zone {zone_id} 업로드 완료! (URL: https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key})")

    print("\n=== 모든 구역 원본 영상 업로드 완료 ===")

if __name__ == "__main__":
    upload_sources()
