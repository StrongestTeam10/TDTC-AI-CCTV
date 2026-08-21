"""
server/config.py - 경로, 환경 변수, DB 설정 및 전역 데이터/Pydantic 모델 정의
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================================================================
# 경로 설정
# =========================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CCTV_UPLOAD_DIR = os.path.join(BASE_DIR, "cctv_upload")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_UPLOAD_DIR = os.path.join(RESULTS_DIR, "temp_uploads")
UPLOAD_DIR = os.path.join(BASE_DIR, "cctv_upload", "uploads")
CACHE_DIR = os.path.join(BASE_DIR, "cctv_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.append(BASE_DIR)
sys.path.append(CCTV_UPLOAD_DIR)
sys.path.append(os.path.join(BASE_DIR, "cctv_ai_pipeline"))
sys.path.append(os.path.join(BASE_DIR, "cctv_ai_pipeline", "sensor_fusion_archive"))

# =========================================================================
# 환경 변수 및 프로필(Profile: dev / prod) 동적 로드
# =========================================================================
active_profile = (os.environ.get("APP_ENV") or os.environ.get("PROFILE") or "prod").lower()

env_files_to_try = [
    os.path.join(BASE_DIR, f".env.{active_profile}"),
    os.path.join(BASE_DIR, ".env")
]

loaded_file = None
for ef in env_files_to_try:
    if os.path.exists(ef):
        load_dotenv(ef, override=True)
        loaded_file = ef
        break

print(f"[Config] [Profile: {active_profile.upper()}] Loaded: {os.path.basename(loaded_file) if loaded_file else '.env'}")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME", "tdtc-cctv-upload")
PYTHON_EXE = sys.executable

# =========================================================================
# DB 커넥터 임포트 (모듈 레벨 - 경로 보정 후 즉시 로드)
# =========================================================================
try:
    # pyrefly: ignore [missing-import]
    from utils.db_connector import bulk_insert_pedestrian_coordinate_json
    db_available = True
    print("[AI Server Config] db_connector 모듈 로드 성공")
except ImportError as _db_import_err:
    db_available = False
    bulk_insert_pedestrian_coordinate_json = None
    print(f"[AI Server Config WARNING] db_connector 로드 실패 - DB 적재 비활성화: {_db_import_err}")

# =========================================================================
# 전역 분석 상태 관리
# =========================================================================
# subprocess 방식 상태 (기존 /api/analyze/* 엔드포인트용)
analysis_state: Dict[str, Any] = {
    "status": "idle",           # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "message": "대기 중",
    "result_count": 0,
    "progress_percent": 0.0,
    "error": None,
}

# WebSocket 스트리밍 방식 상태 (/api/v1/cctv/* 엔드포인트용)
pipeline_state: Dict[str, Any] = {
    "is_analyzing": False,
    "current_video": None,
    "frame_id": 0,
    "pedestrian_count": 0,
    "cri_score": 0.0,
    "status": "IDLE",
}

# =========================================================================
# Pydantic 요청/응답 모델
# =========================================================================
class AnalyzeTriggerRequest(BaseModel):
    start_time: Optional[str] = None
    fps: Optional[float] = 10.0
    test_run: Optional[bool] = False


class AlertTriggerRequest(BaseModel):
    zone_id: int
    alert_type: Optional[str] = "CROWD_CRITICAL"


class AnalyzeStatusResponse(BaseModel):
    status: str
    started_at: Optional[str]
    finished_at: Optional[str]
    message: str
    result_count: int
    progress_percent: float
    error: Optional[str]


class ConfirmRequest(BaseModel):
    zone_id: int
    alert_type: Optional[str] = "CROWD_CRITICAL"
    timestamp_sec: Optional[float] = None
    video_filename: Optional[str] = None
    llm_summary: Optional[str] = "실시간 지능형 CCTV 분석에 의해 인파 밀집 위험이 감지되어 현장 관제실 출동이 확정되었습니다."
    video_id: Optional[int] = 1
