# 00_run_cctv_pipeline.py
# CCTV AI 분석 전체 파이프라인 (01 -> 02) 단독 통합 실행 스크립트입니다.

import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON_EXE = sys.executable

# 1. 입력 비디오 및 설정
INPUT_VIDEO = os.environ.get("INPUT_VIDEO", r"E:\test\cctv_cafe_output.mp4")
ZONE_ID = os.environ.get("ZONE_ID", "1")
TARGET_FPS = os.environ.get("TARGET_FPS", "10")

# 2. 결과물 경로
CCTV_BEV_CSV = os.path.join(RESULTS_DIR, f"cctv_bev_coordinates_zone_{ZONE_ID}.csv")
ANONYMIZED_MP4 = os.path.join(RESULTS_DIR, f"cctv_anonymized_zone_{ZONE_ID}.mp4")

print("=" * 60)
print(f"🚀 [TDTC-AI-CCTV] 00단계 전체 파이프라인 수동/단독 실행 개시")
print(f"   입력 비디오 : {INPUT_VIDEO}")
print(f"   구역 ID    : {ZONE_ID}")
print(f"   Target FPS : {TARGET_FPS}")
print("=" * 60)

# [Step 1] 01_video_to_bev_anonymized.py 실행 (YOLO 모자이크 & CSRNet BEV 좌표 추출)
script_01 = os.path.join(BASE_DIR, "01_video_to_bev_anonymized.py")
env_01 = os.environ.copy()
env_01.update({
    "OUTPUT_MP4": INPUT_VIDEO,
    "CCTV_BEV_CSV": CCTV_BEV_CSV,
    "CSR_RESULT_MP4": ANONYMIZED_MP4,
    "ZONE_ID": str(ZONE_ID),
    "TARGET_FPS": str(TARGET_FPS),
    "PYTHONIOENCODING": "utf-8"
})

print(f"\n▶️ [Step 1/2] 01_video_to_bev_anonymized.py 실행 중...")
start_time = time.time()
res_01 = subprocess.run([PYTHON_EXE, script_01], env=env_01)
if res_01.returncode != 0:
    print(f"❌ [Step 1 실패] 종료 코드: {res_01.returncode}")
    sys.exit(res_01.returncode)
print(f"✅ [Step 1 완료] 소요시간: {time.time() - start_time:.2f}초")

# [Step 2] 02_aggregate_pedestrian_json.py 실행 (좌표 집계 & JSON 생성)
script_02 = os.path.join(BASE_DIR, "02_aggregate_pedestrian_json.py")
env_02 = os.environ.copy()
env_02.update({
    "INPUT_PEDESTRIAN_CSV": CCTV_BEV_CSV,
    "PYTHONIOENCODING": "utf-8"
})

print(f"\n▶️ [Step 2/2] 02_aggregate_pedestrian_json.py 실행 중...")
start_time = time.time()
res_02 = subprocess.run([PYTHON_EXE, script_02], env=env_02)
if res_02.returncode != 0:
    print(f"❌ [Step 2 실패] 종료 코드: {res_02.returncode}")
    sys.exit(res_02.returncode)
print(f"✅ [Step 2 완료] 소요시간: {time.time() - start_time:.2f}초")

print("\n" + "=" * 60)
print("🎉 [TDTC-AI-CCTV] 전체 파이프라인 수행 완료!")
print(f"   결과 CSV   : {CCTV_BEV_CSV}")
print(f"   비식별화MP4: {ANONYMIZED_MP4}")
print("=" * 60)
