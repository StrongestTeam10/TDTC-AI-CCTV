"""
ai_server.py - CCTV AI 분석 FastAPI 서버
=====================================
기존 coordinator.py 파이프라인을 HTTP API로 래핑합니다.

실행 방법:
    uvicorn ai_server:app --host 0.0.0.0 --port 8000 --reload

엔드포인트:
    GET  /health                   - 서버 상태 확인
    POST /api/analyze/trigger      - 분석 파이프라인 실행 (비동기 백그라운드)
    GET  /api/analyze/status       - 현재 분석 진행 상태 조회
    POST /api/alerts/trigger       - Java 백엔드로 긴급 알람 전송
    GET  /api/results/latest       - 최근 분석 결과 조회
"""

import os
import sys
import json
import requests
import asyncio
import subprocess
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================================
# 경로 설정
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CCTV_UPLOAD_DIR = os.path.join(BASE_DIR, "cctv_upload")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 파이프라인 모듈 경로 추가
sys.path.append(BASE_DIR)
sys.path.append(CCTV_UPLOAD_DIR)
sys.path.append(os.path.join(BASE_DIR, "cctv_ai_pipeline"))
sys.path.append(os.path.join(BASE_DIR, "cctv_ai_pipeline", "sensor_fusion_archive"))

# =========================================================================
# 환경 변수 로드
# =========================================================================
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "tdtc-super-secret-key-2026")
PYTHON_EXE = sys.executable

# =========================================================================
# 전역 분석 상태 관리
# =========================================================================
analysis_state = {
    "status": "idle",           # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "message": "대기 중",
    "result_count": 0,
    "progress_percent": 0.0,    # 0.0 ~ 100.0% 실시간 진행률
    "error": None,
}


# =========================================================================
# Pydantic 요청/응답 모델
# =========================================================================
class AnalyzeTriggerRequest(BaseModel):
    start_time: Optional[str] = None       # 예: "2026-08-06 14:00:00"
    fps: Optional[float] = 10.0
    test_run: Optional[bool] = False       # True면 첫 번째 영상만 테스트 분석


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


# 업로드 임시 디렉터리 생성
TEMP_UPLOAD_DIR = os.path.join(RESULTS_DIR, "temp_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# =========================================================================
# 백그라운드 분석 실행 함수
# =========================================================================
def run_pipeline_background(video_path: str, start_time: str, fps: float, zone_id: int):
    """coordinator.py 또는 개별 분석 스크립트를 임시 업로드된 영상에 대해 실행하며 실시간 진행률을 추적"""
    global analysis_state
    import re

    analysis_state.update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "message": "CCTV AI 업로드 비디오 분석 실행 중...",
        "result_count": 0,
        "progress_percent": 0.0,
        "error": None,
    })

    try:
        # 단일 업로드 비디오에 대해 분석을 트리거하기 위해,
        # coordinator.py가 인식하는 환경변수 또는 개별 04/09단계 스크립트를 수동 빌드하여 구동
        steps_dir = os.path.join(CCTV_UPLOAD_DIR, "steps")
        if not os.path.exists(steps_dir):
            steps_dir = os.path.join(BASE_DIR, "cctv_ai_pipeline")

        # 1. 04단계 비디오 -> BEV CSV 추출
        csv_path = os.path.join(RESULTS_DIR, f"temp_bev_zone_{zone_id}.csv")
        script_04 = os.path.join(steps_dir, "04_video_to_bev_CSR.py")
        if not os.path.exists(script_04):
            script_04 = os.path.join(BASE_DIR, "cctv_upload", "core", "04_video_to_bev_CSR.py")

        env = os.environ.copy()
        env.update({
            "OUTPUT_MP4": video_path,
            "CCTV_BEV_CSV": csv_path,
            "CSRNET_MODEL_PATH": os.path.join(BASE_DIR, "models", "csrnet_ultimate_epoch_8.pth"),
            "DATASET_TYPE": "MALL",
            "ZONE_ID": str(zone_id),
            "TARGET_FPS": str(fps),
            "PYTHONIOENCODING": "utf-8"
        })

        print(f"[AI SERVER] 04단계 실행 시작: {video_path}")
        analysis_state["message"] = "Step 1: AI 추론 및 물리 좌표 매핑 중 (0% ~ 80%)"
        
        # Popen을 사용하여 실시간 tqdm 출력 감시
        proc_04 = subprocess.Popen(
            [PYTHON_EXE, script_04], 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding="utf-8",
            bufsize=1
        )

        # tqdm 출력에서 퍼센트 추출하는 정규식 패턴 (예: " 45%|████|")
        percent_pattern = re.compile(r"(\d+)%")

        # stderr를 라인 단위(또는 버퍼 단위)로 지연 없이 읽어 처리
        while True:
            # readline은 tqdm의 \r 출력을 온전하게 받아올 수 있도록 \r, \n 모두 줄끝으로 인식합니다.
            line = proc_04.stderr.readline()
            if not line and proc_04.poll() is not None:
                break
            
            if line:
                match = percent_pattern.search(line)
                if match:
                    tqdm_percent = float(match.group(1))
                    # Step 1 구간 진행률은 전체의 80%로 스케일링 (0% ~ 80%)
                    scaled_percent = round(tqdm_percent * 0.8, 1)
                    analysis_state["progress_percent"] = scaled_percent
                    analysis_state["message"] = f"Step 1: AI 추론 및 물리 좌표 매핑 중 ({scaled_percent}%)"

        proc_04.wait()
        if proc_04.returncode != 0:
            stderr_err = proc_04.stderr.read() or "04단계 스크립트 실행 중 에러가 발생했습니다."
            raise RuntimeError(f"BEV 좌표 추출(04단계) 실패: {stderr_err[-500:]}")

        analysis_state["progress_percent"] = 80.0
        analysis_state["message"] = "Step 2: 보행자 추적 및 DB 적재 중 (80% ~ 95%)"

        # 2. 09단계 CSV -> Supabase 적재 및 JSON 집계
        clip_id = 999  # 임시 수동 분석 클립 ID
        s3_url = f"https://mangwon-cctv.s3.ap-northeast-2.amazonaws.com/danger-clips/uploaded_clip_{zone_id}.mp4"
        script_09 = os.path.join(steps_dir, "09_aggregate_pedestrian_json.py")
        
        env_09 = os.environ.copy()
        env_09.update({
            "INPUT_PEDESTRIAN_CSV": csv_path,
            "CLIP_ID": str(clip_id),
            "ZONE_ID": str(zone_id),
            "S3_CLIP_URL": s3_url,
            "START_TIME": start_time,
            "FPS": str(fps),
            "SKIP_DB_INSERT": "FALSE",
            "PYTHONIOENCODING": "utf-8"
        })

        print(f"[AI SERVER] 09단계 집계 및 DB 적재 실행 시작")
        proc_09 = subprocess.Popen(
            [PYTHON_EXE, script_09], 
            env=env_09, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding="utf-8",
            bufsize=1
        )

        # 09단계는 비교적 짧으므로 처리 단계에 맞춰 80% ~ 95% 구간 수동 업데이트
        while True:
            line = proc_09.stdout.readline()
            if not line and proc_09.poll() is not None:
                break
            if "필터링 완료" in line or "ROI 적용" in line:
                analysis_state["progress_percent"] = 85.0
                analysis_state["message"] = "Step 2: ROI 영역 필터링 완료 (85%)"
            elif "글로벌 추적" in line:
                analysis_state["progress_percent"] = 90.0
                analysis_state["message"] = "Step 2: 객체 추적 및 경로 분석 중 (90%)"
            elif "DB 적재 시도 중" in line:
                analysis_state["progress_percent"] = 95.0
                analysis_state["message"] = "Step 2: Supabase 데이터베이스 적재 중 (95%)"

        proc_09.wait()
        if proc_09.returncode != 0:
            stderr_err = proc_09.stderr.read() or "09단계 스크립트 실행 중 에러가 발생했습니다."
            raise RuntimeError(f"보행자 집계(09단계) 실패: {stderr_err[-500:]}")

        # 임시 CSV 및 업로드 비디오 정리
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

        # 집계된 레코드 수 세기
        result_count = 0
        pedaggr_json = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
        if os.path.exists(pedaggr_json):
            try:
                with open(pedaggr_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result_count = len(data)
            except Exception:
                pass

        analysis_state.update({
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": f"업로드 비디오 분석 완료! {result_count}개 프레임 데이터 DB 적재 성공",
            "result_count": result_count,
            "progress_percent": 100.0,
            "error": None,
        })

    except Exception as e:
        analysis_state.update({
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": "AI 분석 처리 중 에러 발생",
            "error": str(e),
        })


# =========================================================================
# FastAPI 앱 생성
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 CCTV AI 분석 FastAPI 서버 가동!")
    print(f"   백엔드 URL: {BACKEND_URL}")
    print(f"   결과 저장 경로: {RESULTS_DIR}")
    print("=" * 60)
    yield
    print("🛑 서버 종료")


app = FastAPI(
    title="CCTV AI 분석 서버",
    description="CSRNet 기반 CCTV 보행자 분석 파이프라인 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (프론트엔드 / Java 백엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 실제 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# 엔드포인트
# =========================================================================

@app.get("/health", tags=["상태"])
def health_check():
    """서버 및 GPU 상태 확인"""
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "없음"
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "available": gpu_available,
            "device_name": gpu_name,
        },
        "pipeline_status": analysis_state["status"],
    }


@app.post("/api/analyze/trigger", tags=["분석"])
async def trigger_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    zone_id: int = Form(1),
    start_time: Optional[str] = Form(None),
    fps: float = Form(10.0),
):
    """
    CCTV MP4 비디오 파일을 직접 업로드하여 백그라운드 AI 분석을 트리거합니다.
    - 업로드된 파일은 임시 저장된 후 모델 추론 완료 즉시 자동 소멸됩니다.
    - 추출된 좌표 및 위험 데이터는 Supabase DB에 실시간 적재됩니다.
    """
    if analysis_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="이미 분석이 진행 중입니다. /api/analyze/status에서 현재 상태를 확인하세요."
        )

    # 1. 업로드 파일 임시 저장
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_name = f"upload_{timestamp_str}_{file.filename}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_file_name)

    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임시 파일 업로드 실패: {e}")

    analysis_start_time = start_time or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # 2. 백그라운드 분석 실행
    background_tasks.add_task(
        run_pipeline_background,
        video_path=temp_file_path,
        start_time=analysis_start_time,
        fps=fps,
        zone_id=zone_id
    )

    return {
        "message": "✅ 비디오 업로드 성공 및 AI 분석 파이프라인 백그라운드 구동 시작",
        "uploaded_filename": file.filename,
        "zone_id": zone_id,
        "start_time": analysis_start_time,
        "fps": fps,
        "status_url": "/api/analyze/status",
    }


@app.get("/api/analyze/status", response_model=AnalyzeStatusResponse, tags=["분석"])
def get_analysis_status():
    """현재 분석 파이프라인 진행 상태를 반환합니다."""
    return AnalyzeStatusResponse(**analysis_state)


@app.post("/api/alerts/trigger", tags=["알람"])
def trigger_alert(request: AlertTriggerRequest):
    """
    Java 백엔드 /api/ai/alerts/trigger를 호출하여 긴급 알람을 발생시킵니다.
    (Python 파이프라인이 위험 감지 시 직접 호출하는 용도)
    """
    url = f"{BACKEND_URL}/api/ai/alerts/trigger"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": BACKEND_API_KEY,
    }
    payload = {
        "zoneId": request.zone_id,
        "alertType": request.alert_type,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return {"message": "✅ 긴급 알람이 Java 백엔드로 전송됐습니다.", "backend_response": response.text}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Java 백엔드 오류: {response.text}"
            )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Java 백엔드({BACKEND_URL})에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
        )


@app.get("/api/results/latest", tags=["결과"])
def get_latest_results(limit: int = 10):
    """
    최근 분석 결과 (pedaggr01h_full_dataset.json)에서 최신 N개 프레임을 반환합니다.
    """
    pedaggr_json = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
    if not os.path.exists(pedaggr_json):
        return {"message": "분석 결과 파일이 없습니다. 먼저 /api/analyze/trigger를 호출하세요.", "data": []}

    try:
        with open(pedaggr_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "total": len(data),
            "returned": min(limit, len(data)),
            "data": data[-limit:],  # 최신 N개
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 파일 읽기 실패: {str(e)}")


# =========================================================================
# 직접 실행 시
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8088, reload=True)
