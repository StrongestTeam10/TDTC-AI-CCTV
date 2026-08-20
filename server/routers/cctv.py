"""
server/routers/cctv.py - CCTV 비디오 업로드, 스트리밍, 결과 다운로드 라우터
"""

import json
import os
import cv2
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict
from fastapi import APIRouter, BackgroundTasks, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np

from server.config import (
    BASE_DIR, TEMP_UPLOAD_DIR, UPLOAD_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR, pipeline_state
)
from server.s3_uploader import download_file_from_s3
from server.models import (
    torch_available, CSRNet, csr_transform, extract_peaks_from_density, apply_mosaic
)
from server.services import process_ai_pipeline
from server.websocket import manager
from server.video_buffer import global_buffer_manager
from server.raw_video_worker import RawVideoSplitter

router = APIRouter(tags=["분석 (WebSocket)"])

# 비전 AI 모델 전역 싱글톤 캐시
_vision_models = {
    "csrnet": None,
    "yolo": None,
    "hog": None,
    "device": None,
    "initialized": False
}

# 3개 구역의 최신 상태를 전역으로 보관 (존별 값 보존 및 대시보드 전체 합계 실시간 동기화)
_latest_live_zone_metrics: Dict[int, dict] = {
    1: {"pedestrian_count": 0, "occupancy_rate": 0.0, "stagnation_sec": 0.0, "cri_score": 10.0, "risk_level": "SAFE"},
    2: {"pedestrian_count": 0, "occupancy_rate": 0.0, "stagnation_sec": 0.0, "cri_score": 10.0, "risk_level": "SAFE"},
    3: {"pedestrian_count": 0, "occupancy_rate": 0.0, "stagnation_sec": 0.0, "cri_score": 10.0, "risk_level": "SAFE"},
}
_global_live_frame_seq: int = 1
_zone_splitters: Dict[int, RawVideoSplitter] = {}
_broadcast_task: Optional[asyncio.Task] = None


async def _central_broadcast_loop():
    """3개 구역의 최신 상태를 0.2초(5 FPS)마다 완벽하게 단일 묶음으로 동기화 브로드캐스트 (대시보드 튐 100% 원천 차단)"""
    global _global_live_frame_seq
    while True:
        try:
            await asyncio.sleep(0.2)
            _global_live_frame_seq = (_global_live_frame_seq % 100000) + 1
            curr_sync_frame_id = _global_live_frame_seq
            now_ts = datetime.now(timezone.utc).timestamp()

            for z_id in (1, 2, 3):
                z_data = _latest_live_zone_metrics.get(z_id, {
                    "pedestrian_count": 0, "occupancy_rate": 0.0, "stagnation_sec": 0.0, "cri_score": 10.0, "risk_level": "SAFE"
                })
                await manager.broadcast({
                    "type": "CCTV_AI_STREAM",
                    "frame_id": curr_sync_frame_id,
                    "zone_id": z_id,
                    "filename": f"zone_{z_id}_stream.mp4",
                    "pedestrian_count": z_data["pedestrian_count"],
                    "occupancy_rate": z_data["occupancy_rate"],
                    "stagnation_sec": z_data["stagnation_sec"],
                    "cri_score": z_data["cri_score"],
                    "risk_level": z_data["risk_level"],
                    "timestamp": now_ts
                })
        except Exception:
            pass


def ensure_central_broadcast_started():
    """중앙 동기화 브로드캐스터 태스크 활성화 보장"""
    global _broadcast_task
    if _broadcast_task is None or _broadcast_task.done():
        try:
            loop = asyncio.get_running_loop()
            _broadcast_task = loop.create_task(_central_broadcast_loop())
        except RuntimeError:
            pass


def get_zone_splitter(zone_id: int, fps: float = 10.0) -> RawVideoSplitter:
    """구역당 단일 싱글톤 Splitter 워커 반환 (다중 연결 시 파일 쓰기 충돌 방지)"""
    if zone_id not in _zone_splitters:
        _zone_splitters[zone_id] = RawVideoSplitter(zone_id=zone_id, fps=fps, split_duration_sec=60.0)
    return _zone_splitters[zone_id]



def get_shared_vision_models():
    """실시간 스트리밍을 위한 비전 모델 싱글톤 초기화"""
    if _vision_models["initialized"]:
        return _vision_models

    # 1. PyTorch CSRNet
    if torch_available and CSRNet is not None:
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CSRNet().to(dev)
            m_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
            if os.path.exists(m_path):
                ckpt = torch.load(m_path, map_location=dev)
                sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(sd, strict=False)
                model.eval()
                _vision_models["csrnet"] = model
                _vision_models["device"] = dev
                print(f"[Vision Setup] 실시간 CSRNet 로드 완료 ({dev})")
        except Exception as e:
            print(f"[Vision Setup] CSRNet 로드 생략: {e}")

    # 2. YOLO11n
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(MODELS_DIR, "yolo11n.pt")
        if not os.path.exists(yolo_path):
            yolo_path = os.path.join(MODELS_DIR, "bestYOLOm5080model.pt")
        if not os.path.exists(yolo_path):
            yolo_path = "yolo11n.pt"
        yolo_model = YOLO(yolo_path)
        _vision_models["yolo"] = yolo_model
        print(f"[Vision Setup] 실시간 YOLO11n 로드 완료: {yolo_path}")
    except Exception as e:
        print(f"[Vision Setup] YOLO11n 로드 생략: {e}")

    # 3. HOG
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _vision_models["hog"] = hog
    except Exception:
        pass

    _vision_models["initialized"] = True
    return _vision_models


def find_sample_video_for_zone(zone_id: int) -> Optional[str]:
    """
    구역별(zone_id 1, 2, 3) 실시간 스트림용 원본 비디오 경로 탐색 및 S3 자동 캐싱:
    1. 로컬 캐시(cctv_cache/zone{zone_id}_source.mp4) 최우선 확인
    2. 로컬 캐시가 없으면 AWS S3(source-videos/zone{zone_id}_source.mp4)에서 자동 다운로드
    3. 로컬 zone/ 폴더 또는 results/ 폴더 fallback 탐색
    """
    # 1. 로컬 캐시 확인
    cached_path = os.path.join(CACHE_DIR, f"zone{zone_id}_source.mp4")
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
        return cached_path

    # 2. AWS S3에서 원본 비디오 자동 다운로드
    s3_key = f"source-videos/zone{zone_id}_source.mp4"
    print(f"[Zone {zone_id} Stream] 로컬 캐시 없음 -> S3 ({s3_key}) 자동 다운로드 시도...")
    download_success = download_file_from_s3(s3_key, cached_path)
    if download_success and os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
        print(f"[Zone {zone_id} Stream] S3 원본 비디오 캐싱 완료: {cached_path}")
        return cached_path

    # 3. 로컬 zone/zone_id{zone_id} 폴더 fallback 탐색
    candidates_zone_dirs = [
        os.path.join(BASE_DIR, "..", "zone", f"zone_id{zone_id}"),
        os.path.join(BASE_DIR, "zone", f"zone_id{zone_id}"),
        f"e:/AIVLE_10team/zone/zone_id{zone_id}",
    ]
    for z_dir in candidates_zone_dirs:
        if os.path.exists(z_dir):
            for fname in sorted(os.listdir(z_dir)):
                if fname.lower().endswith(".mp4"):
                    v_path = os.path.join(z_dir, fname)
                    return v_path

    # 4. results 폴더 내 구역 파일 fallback 탐색
    candidates = [
        os.path.join(RESULTS_DIR, f"zone_{zone_id}_live.mp4"),
        os.path.join(RESULTS_DIR, "stabilization_720p_fp16_verification.mp4"),
        os.path.join(RESULTS_DIR, "stabilization_verification.mp4"),
        os.path.join(RESULTS_DIR, "processed_cctv_20260811_223259_test_north04.mp4"),
        os.path.join(RESULTS_DIR, "cctv_simulation_video.mp4"),
        os.path.join(BASE_DIR, "cctv_simulation_video.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


async def generate_mjpeg_stream(zone_id: int):
    """실시간 AI 보행자 검출 + 가우시안 모자이크 비식별화 MJPEG 스트림 생성기"""
    video_path = find_sample_video_for_zone(zone_id)
    if not video_path:
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"CCTV Zone {zone_id} Stream Waiting...", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.1)

    v_models = get_shared_vision_models()
    csrnet = v_models["csrnet"]
    yolo = v_models["yolo"]
    hog = v_models["hog"]
    device = v_models["device"]

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1800
    if orig_fps < 10 or orig_fps > 120:
        orig_fps = 30.0
    
    # ⭐️ [벽시계 기준 1.0x 정속 재생]
    # 예전에는 "한 바퀴 = 원본 frame_step 프레임 진행"으로 고정돼 있어서, AI 처리
    # 시간(elapsed)이 프레임 간격(33ms)을 넘는 순간부터 따라잡을 방법이 없어
    # 재생이 슬로모션이 됐다(MJPEG는 타임스탬프가 없어 서버 송출 속도가 곧 재생 속도다).
    # 이제 스트림 시작 시각을 기준으로 "지금 보여줘야 할 원본 프레임 번호"를 계산해
    # 밀린 만큼 건너뛴다. 처리가 느리면 화면 fps가 떨어질 뿐 배속은 1.0x로 유지된다.
    import time
    target_stream_fps = 30.0
    frame_interval = 1.0 / target_stream_fps
    stream_start = time.time()
    consumed_frames = 0  # 지금까지 원본에서 소비(grab/read)한 프레임 수. 영상 루프와 무관하게 단조 증가.

    frame_idx = 0
    cached_boxes = []
    cached_peaks = []

    smoothed_count = 0.0
    smoothed_stagnation = 0.0
    smoothed_cri = 0.0

    # 35초 위험 클립 추출용 원형 버퍼 및 1분 단위 상시 영상 분할 워커 초기화
    buf = global_buffer_manager.get_buffer(zone_id)
    splitter = get_zone_splitter(zone_id, fps=target_stream_fps)

    try:
        while True:
            t_start = time.time()

            # 벽시계 기준으로 "지금 보여줘야 할 원본 프레임"까지 건너뛴다.
            due_frames = int((t_start - stream_start) * orig_fps) + 1
            advance = due_frames - consumed_frames
            if advance < 1:
                advance = 1
            # 스톨(일시 정지·모델 로딩 등) 직후 수백 프레임을 한꺼번에 grab하며 폭주하지
            # 않도록 상한(2초치)을 두고, 더 밀린 시간은 기준 시각을 옮겨 탕감한다.
            max_advance = int(orig_fps * 2)
            if advance > max_advance:
                stream_start += (advance - max_advance) / orig_fps
                advance = max_advance

            for _ in range(advance - 1):
                if not cap.grab():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap.grab()
            consumed_frames += advance

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    # 안전 재시도
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            frame_idx += 1

            # 좌우 검은색 필러박스 크롭 (가로 영상 꽉 차게)
            h, w = frame.shape[:2]
            if w > h and w >= 1900 and h >= 1000:
                frame = frame[:, 656:1264]

            # 4프레임마다 1회 딥러닝 추론 수행 (30 FPS 방어)
            if frame_idx % 4 == 1 or not cached_boxes:
                detected_peaks = []
                detected_boxes = []

                # 1. CSRNet 군중 밀도 분석 & 피크 좌표 검출 (인원수 및 3D 물리 좌표 산출)
                if csrnet is not None and device is not None:
                    try:
                        import torch
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            dm = csrnet(tensor)
                            dm = torch.clamp(dm, min=0)
                        d_map = dm.squeeze().cpu().numpy()
                        h_out, w_out = dm.shape[2], dm.shape[3]
                        scale_x = frame.shape[1] / w_out
                        scale_y = frame.shape[0] / h_out
                        raw_peaks = extract_peaks_from_density(d_map, threshold=0.0004)
                        for rx, ry in raw_peaks:
                            px, py = float(rx * scale_x), float(ry * scale_y)
                            if 20.0 <= px <= (frame.shape[1] - 20.0) and 30.0 <= py <= (frame.shape[0] - 15.0):
                                detected_peaks.append((px, py))
                                # CSRNet 피크 주변 모자이크 박스 기본 생성 (원복)
                                detected_boxes.append((px - 28, py - 32, px + 28, py + 32))
                    except Exception:
                        pass

                # 2. YOLO11n 실시간 보행자 비식별화 박스 검출 (원복)
                if yolo is not None:
                    try:
                        res = yolo(frame, classes=[0], verbose=False, conf=0.25, imgsz=384)
                        if len(res) > 0 and len(res[0].boxes) > 0:
                            for b in res[0].boxes:
                                coords = b.xyxy[0].cpu().numpy()
                                detected_boxes.append((float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])))
                    except Exception:
                        pass

                if detected_peaks:
                    cached_peaks = detected_peaks
                elif detected_boxes:
                    cached_peaks = [(float((b[0]+b[2])/2), float(b[3])) for b in detected_boxes]

                if detected_boxes:
                    cached_boxes = detected_boxes

            # 2. 모자이크 비식별화 즉시 적용
            for (x1, y1, x2, y2) in cached_boxes:
                frame = apply_mosaic(frame, int(x1), int(y1), int(x2), int(y2))

            # 3. 실시간 지표 산출 및 이중 EMA 스무딩
            raw_count = len(active_peaks) if 'active_peaks' in locals() else len(cached_peaks)
            smoothed_count = raw_count if smoothed_count == 0 else (smoothed_count * 0.75 + raw_count * 0.25)
            final_count = int(round(smoothed_count))

            occupancy_rate = min(100.0, round(final_count * 2.2, 1))
            target_stagnation = max(0.0, (final_count - 8) * 0.7) if final_count > 8 else max(0.0, final_count * 0.15)
            smoothed_stagnation = target_stagnation if smoothed_stagnation == 0.0 else (smoothed_stagnation * 0.85 + target_stagnation * 0.15)
            stagnation_sec = round(min(25.0, smoothed_stagnation), 1)

            raw_cri = (final_count * 1.8) + (stagnation_sec * 0.35) + (occupancy_rate * 0.25)
            smoothed_cri = raw_cri if smoothed_cri == 0.0 else (smoothed_cri * 0.8 + raw_cri * 0.2)
            cri_score = round(min(100.0, max(5.0, smoothed_cri)), 1)
            risk_level = "SAFE" if cri_score < 40 else "WARN" if cri_score < 70 else "DANGER"

            # 전역 최신 구역 상태 갱신 (WebSocket 브로드캐스터 연동)
            _latest_live_zone_metrics[zone_id] = {
                "pedestrian_count": final_count,
                "occupancy_rate": occupancy_rate,
                "stagnation_sec": stagnation_sec,
                "cri_score": cri_score,
                "risk_level": risk_level,
            }
            ensure_central_broadcast_started()

            # 프레임별 보행자 좌표 및 AI 지표 패킹
            active_peaks = cached_peaks if cached_peaks else []
            pixels_dict = {f"p_{i+1}": [int(px), int(py)] for i, (px, py) in enumerate(active_peaks)}
            bev_dict = {f"p_{i+1}": [round(float(px * 0.015), 2), round(float(py * 0.02), 2), 0.0] for i, (px, py) in enumerate(active_peaks)}
            frame_metric = {
                "zone_id": zone_id,
                "frame_id": frame_idx,
                "video_id": 1,
                "total_count": len(active_peaks),
                "occupancy_rate": occupancy_rate,
                "stagnation_sec": stagnation_sec,
                "cri_score": cri_score,
                "risk_level": risk_level,
                "pixels_json": json.dumps(pixels_dict),
                "bev_xyz_json": json.dumps(bev_dict),
                "captured_at": datetime.now(timezone.utc).isoformat()
            }

            # 35초 비상 클립 버퍼 및 1분 상시 분할 워커에 프레임 기록
            buf.append_frame(frame, t_start)
            splitter.write_frame(frame, frame_metric)

            # 4. ⭐️ [30 FPS 최적화] 가로 540px + JPEG 품질 68
            fh, fw = frame.shape[:2]
            target_w = 540
            disp_h = int(fh * (float(target_w) / fw))
            disp_frame = cv2.resize(frame, (target_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # 품질 68로 경량 고속 30 FPS 스트림 송출
            _, buffer = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # 30 FPS 정속 유지 및 비동기 스케줄링
            elapsed = time.time() - t_start
            sleep_time = max(0.002, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)
    finally:
        cap.release()


@router.get("/api/v1/cctv/stream")
async def stream_cctv_live(zone_id: int = 1):
    """프론트엔드 갤러리 카드/모달용 실시간 MJPEG 비디오 스트리밍 엔드포인트"""
    return StreamingResponse(
        generate_mjpeg_stream(zone_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/api/v1/cctv/db-status", tags=["진단"])
def get_db_and_splitter_status():
    """현재 Supabase/RDS DB 테이블 적재 건수 및 실시간 1분 분할 워커 진행 상태 반환"""
    from utils.db_connector import get_db_connection
    db_counts = {"vdoclip01m": 0, "pedaggr01h": 0, "mrkrisk01m": 0, "connected": False}
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM vdoclip01m;")
            db_counts["vdoclip01m"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM pedaggr01h;")
            db_counts["pedaggr01h"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM mrkrisk01m;")
            db_counts["mrkrisk01m"] = cur.fetchone()[0]
            cur.close()
            conn.close()
            db_counts["connected"] = True
        else:
            db_counts["error"] = "Database connection returned None (check credentials)"
    except Exception as e:
        db_counts["error"] = str(e)

    splitters_info = {}
    for zid, sp in _zone_splitters.items():
        splitters_info[f"zone_{zid}"] = {
            "frame_count": sp.frame_count,
            "max_frames": sp.max_frames_per_split,
            "progress_pct": round((sp.frame_count / max(1, sp.max_frames_per_split)) * 100, 1),
            "buffered_metrics_count": len(sp.buffered_metrics)
        }

    return {
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_counts,
        "live_splitters": splitters_info
    }


@router.post("/api/v1/cctv/flush-db", tags=["진단"])
def flush_current_splitters_to_db():
    """현재까지 모인 모든 구역의 1분 버퍼를 즉시 S3 및 RDS DB에 강제 마감 적재"""
    flushed = []
    for zid, sp in _zone_splitters.items():
        if sp.frame_count > 0:
            sp.finalize()
            flushed.append(f"zone_{zid} ({sp.frame_count} frames)")
    return {"message": "✅ 강제 DB 적재 트리거 완료", "flushed_zones": flushed}



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
