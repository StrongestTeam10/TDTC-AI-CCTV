"""
server/routers/cctv.py - CCTV 비디오 업로드, 스트리밍, 결과 다운로드 라우터
"""

import os
import cv2
import asyncio
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np

from server.config import (
    BASE_DIR, TEMP_UPLOAD_DIR, UPLOAD_DIR, RESULTS_DIR, pipeline_state
)
from server.services import process_ai_pipeline
from server.websocket import manager

router = APIRouter(tags=["분석 (WebSocket)"])


def find_sample_video_for_zone(zone_id: int) -> Optional[str]:
    """구역별 실시간 스트림용 샘플 비디오 경로 탐색 (고품질 stabilization_verification 우선)"""
    candidates = [
        os.path.join(RESULTS_DIR, f"zone_{zone_id}_live.mp4"),
        os.path.join(RESULTS_DIR, "stabilization_720p_fp16_verification.mp4"),
        os.path.join(RESULTS_DIR, "stabilization_verification.mp4"),
        os.path.join(RESULTS_DIR, "processed_cctv_20260811_223259_test_north04.mp4"),
        os.path.join(RESULTS_DIR, "cctv_simulation_video.mp4"),
        os.path.join(BASE_DIR, "cctv_simulation_video.mp4"),
        "e:/AIVLE_10team/TDTC-AI-FE/dist/cctv_mangwon_live.mp4",
        "e:/AIVLE_10team/ai_pipeline/results/stabilization_720p_fp16_verification.mp4",
        "e:/AIVLE_10team/ai_pipeline/results/stabilization_verification.mp4",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


async def generate_mjpeg_stream(zone_id: int):
    """MJPEG 스트림 생성기 (무한 루프)"""
    video_path = find_sample_video_for_zone(zone_id)
    if not video_path:
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"CCTV Zone {zone_id} LIVE STREAM", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (450, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.1)

    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            if frame.shape[0] != 720 or frame.shape[1] != 1280:
                frame = cv2.resize(frame, (1280, 720))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.1) # 10 FPS
    finally:
        cap.release()


@router.get("/api/v1/cctv/stream")
async def stream_cctv_live(zone_id: int = 1):
    """프론트엔드 갤러리 카드/모달용 실시간 MJPEG 비디오 스트리밍 엔드포인트"""
    return StreamingResponse(
        generate_mjpeg_stream(zone_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/api/v1/cctv/upload")
async def upload_cctv_video(
    background_tasks: BackgroundTasks,
    cctv_video: UploadFile = File(...),
    zone_id: int = Form(1),
    start_time: Optional[str] = Form(None),
    fps: float = Form(10.0),
):
    """
    CCTV 비디오 업로드 → CSRNet 추론 + 모자이크 처리 + WebSocket 실시간 스트리밍 및 Supabase DB 최종 적재.
    분석 진행 상황은 WS /ws/cctv-stream 으로 수신합니다.
    """
    if pipeline_state["is_analyzing"]:
        raise HTTPException(status_code=409, detail="이미 WebSocket 분석이 진행 중입니다.")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_file_name = f"cctv_{timestamp_str}_{cctv_video.filename}"
    saved_file_path = os.path.join(UPLOAD_DIR, saved_file_name)

    try:
        with open(saved_file_path, "wb") as buffer:
            buffer.write(await cctv_video.read())
        print(f"[CCTV Upload] 비디오 업로드 완료: {saved_file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CCTV 비디오 파일 저장 실패: {e}")

    analysis_start_time = start_time or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    pipeline_state["is_analyzing"] = True
    pipeline_state["current_video"] = saved_file_name
    pipeline_state["frame_id"] = 0
    pipeline_state["pedestrian_count"] = 0
    pipeline_state["cri_score"] = 0.0
    pipeline_state["status"] = "ANALYZING"

    background_tasks.add_task(
        process_ai_pipeline, 
        saved_file_path, 
        saved_file_name, 
        zone_id, 
        analysis_start_time, 
        fps
    )

    return {
        "status": "SUCCESS",
        "message": "✅ CCTV 동영상 파일 업로드 성공 및 실시간 AI 스트리밍 파이프라인 개시",
        "video_info": {
            "filename": saved_file_name,
            "original_name": cctv_video.filename,
            "zone_id": zone_id,
            "start_time": analysis_start_time,
            "target_fps": fps,
            "file_path": saved_file_path
        },
        "websocket_endpoint": "/ws/cctv-stream",
        "video_result_url": f"/api/v1/cctv/video/{saved_file_name}",
        "dataset_result_url": f"/api/v1/cctv/dataset/{saved_file_name}"
    }


@router.get("/api/v1/cctv/status")
def get_cctv_pipeline_status():
    """파이프라인 진행 상태 조회"""
    return {
        "status": "OK",
        "pipeline_state": pipeline_state,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/api/v1/cctv/video/{filename}")
async def get_cctv_result_video(filename: str):
    """모자이크 처리된 결과 영상 다운로드 및 웹 스트리밍 반환"""
    video_path = os.path.join(RESULTS_DIR, f"processed_{filename}")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    video_path = os.path.join(RESULTS_DIR, "cctv_result_upload.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="결과 영상 파일이 없습니다.")


@router.get("/api/v1/cctv/dataset/{filename}", tags=["결과"])
async def get_cctv_result_dataset(filename: str):
    """분석 데이터셋 JSON 다운로드"""
    dataset_path = os.path.join(RESULTS_DIR, f"uploaded_{filename}_dataset.json")
    if os.path.exists(dataset_path):
        return FileResponse(dataset_path, media_type="application/json")
    fallback = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
    if os.path.exists(fallback):
        return FileResponse(fallback, media_type="application/json")
    raise HTTPException(status_code=404, detail="분석 데이터셋이 없습니다.")


@router.websocket("/ws/cctv-stream")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 프레임 관제 데이터 WebSocket 스트리밍"""
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "INIT_STATE",
            "pipeline_state": pipeline_state
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "time": datetime.now(timezone.utc).timestamp()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        manager.disconnect(websocket)
