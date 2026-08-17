"""
server/routers/analyze.py - Subprocess 기반 분석 파이프라인 및 출동 확정 라우터
"""

import os
import cv2
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile, Form

from server.config import (
    TEMP_UPLOAD_DIR, UPLOAD_DIR, RESULTS_DIR, BASE_DIR, analysis_state, pipeline_state,
    AnalyzeStatusResponse, ConfirmRequest
)
from server.services import run_pipeline_background, process_confirm_background

router = APIRouter()


@router.post("/api/analyze/trigger", tags=["분석 (subprocess)"])
async def trigger_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    zone_id: int = Form(1),
    start_time: Optional[str] = Form(None),
    fps: float = Form(10.0),
):
    """
    CCTV MP4 비디오 파일 업로드 → subprocess 방식 백그라운드 AI 분석.
    진행률은 GET /api/analyze/status 로 폴링합니다.
    """
    if analysis_state["status"] == "running":
        raise HTTPException(status_code=409, detail="이미 분석이 진행 중입니다. /api/analyze/status 확인.")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_name = f"upload_{timestamp_str}_{file.filename}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_file_name)

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임시 파일 업로드 실패: {e}")

    analysis_start_time = start_time or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    background_tasks.add_task(run_pipeline_background, temp_file_path, analysis_start_time, fps, zone_id)

    return {
        "message": "✅ 비디오 업로드 성공 및 AI 분석 파이프라인 시작",
        "uploaded_filename": file.filename,
        "zone_id": zone_id,
        "start_time": analysis_start_time,
        "fps": fps,
        "status_url": "/api/analyze/status",
    }


@router.get("/api/analyze/status", response_model=AnalyzeStatusResponse, tags=["분석 (subprocess)"])
def get_analysis_status():
    """현재 subprocess 분석 파이프라인 진행 상태 반환 (0~100%)"""
    return AnalyzeStatusResponse(**analysis_state)


@router.post("/api/analyze/confirm", tags=["알람"])
async def confirm_incident(request: ConfirmRequest, background_tasks: BackgroundTasks):
    """
    프론트엔드 출동 버튼 클릭 시 호출됩니다.
    1. 분석 중인 동영상이 있을 경우 해당 타임스탬프 기준 앞뒤 35초 구간을 잘라냅니다.
    2. 사고 명세서 PDF 보고서를 생성합니다.
    3. 생성된 파일들을 S3 버킷에 업로드합니다.
    4. 자바 백엔드로 알람 트리거를 발생시키고 웹훅을 전송합니다.
    """
    video_name = request.video_filename or pipeline_state.get("current_video") or "default_mangwon.mp4"
    video_path = os.path.join(UPLOAD_DIR, video_name)
    
    if not os.path.exists(video_path):
        video_path = os.path.join(TEMP_UPLOAD_DIR, video_name)
        if not os.path.exists(video_path):
            fallback_video = os.path.join(BASE_DIR, "cctv_upload", "mangwon_test.mp4")
            if os.path.exists(fallback_video):
                video_path = fallback_video
            else:
                video_path = None

    cap = None
    fps = 15.0
    total_frames = 600
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    current_sec = request.timestamp_sec
    if current_sec is None:
        curr_frame = pipeline_state.get("frame_id", 0)
        current_sec = curr_frame / fps if curr_frame > 0 else (total_frames / 2) / fps

    start_sec = max(0.0, current_sec - 17.5)
    end_sec = min(total_frames / fps, current_sec + 17.5)
    
    background_tasks.add_task(
        process_confirm_background,
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        zone_id=request.zone_id,
        alert_type=request.alert_type,
        llm_summary=request.llm_summary,
        video_id=request.video_id
    )

    return {
        "status": "SUCCESS",
        "message": "사고 상황 비동기 슬라이싱 및 자바 백엔드 전송 파이프라인 구동 시작",
        "parameters": {
            "target_video": video_name,
            "incident_time_sec": round(current_sec, 2),
            "slice_range": f"{round(start_sec, 2)}s ~ {round(end_sec, 2)}s (35초)"
        }
    }


@router.post("/api/analyze/snapshot", tags=["알람"])
async def trigger_snapshot_webhook(request: ConfirmRequest, background_tasks: BackgroundTasks):
    """
    백엔드/프론트엔드 Webhook용 엔드포인트.
    지정된 구역(Zone)의 최신 35초 위험 클립 및 PDF 명세서를 즉시 생성하여 S3 업로드 및 백엔드 등록을 수행합니다.
    """
    return await confirm_incident(request, background_tasks)
