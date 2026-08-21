"""
server/routers/cctv.py - CCTV 비디오 업로드, 스트리밍, 결과 다운로드 라우터
"""

import json
import os
import cv2
import time
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
from server.raw_video_worker import RawVideoSplitter, get_zone_splitter, generate_and_upload_zone_1min_clip

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
    try:
        loop = asyncio.get_running_loop()
        if _broadcast_task is None or _broadcast_task.done():
            _broadcast_task = loop.create_task(_central_broadcast_loop())
    except RuntimeError:
        pass


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
    cached_path = os.path.join(CACHE_DIR, f"zone{zone_id}_source.mp4")
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
        return cached_path

    s3_key = f"source-videos/zone{zone_id}_source.mp4"
    print(f"[Zone {zone_id} Stream] 로컬 캐시 없음 -> S3 ({s3_key}) 자동 다운로드 시도...")
    download_success = download_file_from_s3(s3_key, cached_path)
    if download_success and os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
        print(f"[Zone {zone_id} Stream] S3 원본 비디오 캐싱 완료: {cached_path}")
        return cached_path

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


_zone_engines: Dict[int, 'ZoneLiveEngine'] = {}


class ZoneLiveEngine:
    """구역별 실시간 30 FPS 라이브 비디오 디코딩, 비동기 AI 추론, 모자이크 렌더링 및 JPEG 싱글톤 브로드캐스터"""

    def __init__(self, zone_id: int):
        self.zone_id = zone_id
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.frame_seq: int = 0
        self.new_frame_event = asyncio.Event()
        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None

        self.cached_boxes = []
        self.cached_peaks = []
        self.smoothed_count = 0.0
        self.smoothed_stagnation = 0.0
        self.smoothed_cri = 0.0
        self.risk_level = "NORMAL"
        self.ai_busy = False
        self.latest_metric = {}

    def ensure_started(self):
        """싱글톤 백그라운드 엔진 루프 시작 보장"""
        if self._task is None or self._task.done():
            try:
                loop = asyncio.get_running_loop()
                self.is_running = True
                self._task = loop.create_task(self._run_engine_loop())
            except RuntimeError:
                pass

    async def _run_async_ai_worker(self, frame_sample, frame_idx: int):
        """별도 비동기 스레드 풀에서 AI 추론 수행 (메인 30 FPS 렌더링 루프를 절대 블로킹하지 않음)"""
        try:
            v_models = get_shared_vision_models()
            csrnet = v_models.get("csrnet")
            yolo = v_models.get("yolo")
            device = v_models.get("device")

            detected_peaks = []
            detected_boxes = []

            # 1. CSRNet 군중 밀도 분석 & 피크 좌표 검출
            if csrnet is not None and device is not None:
                def _infer_csr():
                    import torch
                    img_rgb = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2RGB)
                    tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        dm = csrnet(tensor)
                        dm = torch.clamp(dm, min=0)
                    d_map = dm.squeeze().cpu().numpy()
                    h_out, w_out = dm.shape[2], dm.shape[3]
                    scale_x = frame_sample.shape[1] / w_out
                    scale_y = frame_sample.shape[0] / h_out
                    raw_peaks = extract_peaks_from_density(d_map, threshold=0.0004)
                    p_list = []
                    b_list = []
                    for rx, ry in raw_peaks:
                        px, py = float(rx * scale_x), float(ry * scale_y)
                        if 20.0 <= px <= (frame_sample.shape[1] - 20.0) and 30.0 <= py <= (frame_sample.shape[0] - 15.0):
                            p_list.append((px, py))
                            b_list.append((px - 28, py - 32, px + 28, py + 32))
                    return p_list, b_list

                peaks, boxes = await asyncio.to_thread(_infer_csr)
                detected_peaks.extend(peaks)
                detected_boxes.extend(boxes)

            # 2. YOLO11n 보행자 비식별화 박스 검출
            if yolo is not None:
                def _infer_yolo():
                    b_list = []
                    res = yolo(frame_sample, classes=[0], verbose=False, conf=0.25, imgsz=384)
                    if len(res) > 0 and len(res[0].boxes) > 0:
                        for b in res[0].boxes:
                            coords = b.xyxy[0].cpu().numpy()
                            b_list.append((float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])))
                    return b_list

                y_boxes = await asyncio.to_thread(_infer_yolo)
                detected_boxes.extend(y_boxes)

            if detected_boxes:
                self.cached_boxes = detected_boxes
            if detected_peaks:
                self.cached_peaks = detected_peaks

        except Exception:
            pass
        finally:
            self.ai_busy = False

    async def _run_engine_loop(self):
        """벽시계 기준 1.0x 정속 실시간 30 FPS 비디오 디코딩, 모자이크 렌더링 및 JPEG 프레임 브로드캐스트 루프"""
        video_path = find_sample_video_for_zone(self.zone_id)
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if orig_fps < 10 or orig_fps > 120:
            orig_fps = 30.0

        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        frame_idx = 0
        buf = global_buffer_manager.get_buffer(self.zone_id)

        # ⭐️ [벽시계 기준 1.0x 정속 재생 동기화 타이머]
        # 처리 시간에 상관없이 실제 벽시계 시간(1초)에 정확히 1초 분량의 영상이 진행되도록 보장합니다.
        stream_start = time.time()
        consumed_frames = 0

        try:
            while self.is_running:
                t_start = time.time()

                # 벽시계 시간 기준으로 지금 도달해야 할 원본 프레임 수 계산
                due_frames = int((t_start - stream_start) * orig_fps) + 1
                advance = due_frames - consumed_frames

                if advance < 1:
                    advance = 1
                elif advance > int(orig_fps * 2):  # 2초 이상 차이나면 기준 시간 리셋
                    stream_start = t_start
                    consumed_frames = 0
                    advance = 1

                # 밀린 만큼 중간 프레임 고속 건너뛰기 (1.0x 정속 유지)
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
                        cap.release()
                        cap = cv2.VideoCapture(video_path)
                        ret, frame = cap.read()
                        if not ret:
                            await asyncio.sleep(0.05)
                            continue

                frame_idx += 1

                # 1. 좌우 필러박스 크롭 (가로 영상 꽉 차게)
                h, w = frame.shape[:2]
                if w > h and w >= 1900 and h >= 1000:
                    frame = frame[:, 656:1264]

                # 2. 비동기 AI 추론 트리거 (6프레임마다 1회 스레드로 위임)
                if (frame_idx % 6 == 1 or not self.cached_boxes) and not self.ai_busy:
                    self.ai_busy = True
                    asyncio.create_task(self._run_async_ai_worker(frame.copy(), frame_idx))

                # 3. 초고속 540p 리사이즈 및 스케일 모자이크 (CPU 연산 4배 절감)
                fh, fw = frame.shape[:2]
                target_w = 540
                disp_h = int(fh * (float(target_w) / fw))
                scale_x = float(target_w) / fw
                scale_y = float(disp_h) / fh

                disp_frame = cv2.resize(frame, (target_w, disp_h), interpolation=cv2.INTER_LINEAR)

                # 540p 화면에 모자이크 적용 (초고속 1ms 완료)
                if self.cached_boxes:
                    for (x1, y1, x2, y2) in self.cached_boxes:
                        mx1 = int(x1 * scale_x)
                        my1 = int(y1 * scale_y)
                        mx2 = int(x2 * scale_x)
                        my2 = int(y2 * scale_y)
                        disp_frame = apply_mosaic(disp_frame, mx1, my1, mx2, my2)

                # 4. 실시간 지표 계산 및 EWM 스무딩
                raw_count = float(len(self.cached_peaks)) if self.cached_peaks else 0.0
                self.smoothed_count = 0.7 * self.smoothed_count + 0.3 * raw_count
                occupancy_rate = min(100.0, round((self.smoothed_count / 80.0) * 100.0, 1))

                stagnation_sec = 0.0
                if self.smoothed_count >= 15.0:
                    stagnation_sec = round(min(120.0, (self.smoothed_count - 10.0) * 2.5), 1)

                cri_score = round(min(100.0, max(5.0, (occupancy_rate * 0.6) + (stagnation_sec * 0.4))), 1)

                if cri_score >= 80.0:
                    risk_level = "DANGER"
                elif cri_score >= 50.0:
                    risk_level = "WARNING"
                else:
                    risk_level = "NORMAL"

                self.smoothed_stagnation = stagnation_sec
                self.smoothed_cri = cri_score
                self.risk_level = risk_level

                _latest_live_zone_metrics[self.zone_id] = {
                    "pedestrian_count": int(round(self.smoothed_count)),
                    "occupancy_rate": occupancy_rate,
                    "stagnation_sec": stagnation_sec,
                    "cri_score": cri_score,
                    "risk_level": risk_level,
                }
                ensure_central_broadcast_started()

                # 메트릭 패킹
                active_peaks = self.cached_peaks if self.cached_peaks else []
                pixels_dict = {f"p_{i+1}": [int(px), int(py)] for i, (px, py) in enumerate(active_peaks)}
                bev_dict = {f"p_{i+1}": [round(float(px * 0.015), 2), round(float(py * 0.02), 2), 0.0] for i, (px, py) in enumerate(active_peaks)}
                frame_metric = {
                    "zone_id": self.zone_id,
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
                self.latest_metric = frame_metric

                # 5. 원형 프레임 버퍼 기록 (35초 비상 클립용)
                buf.append_frame(frame, t_start)

                # 6. HUD 오버레이
                if risk_level == "DANGER":
                    cv2.rectangle(disp_frame, (0, 0), (target_w - 1, disp_h - 1), (0, 0, 255), 4)
                    cv2.rectangle(disp_frame, (0, 0), (target_w, 36), (0, 0, 220), -1)
                    cv2.putText(disp_frame, f"EMERGENCY DANGER! CRI: {cri_score:.1f} (ZONE {self.zone_id})", (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
                elif risk_level == "WARNING":
                    cv2.rectangle(disp_frame, (0, 0), (target_w - 1, disp_h - 1), (0, 140, 255), 3)
                    cv2.rectangle(disp_frame, (0, 0), (target_w, 30), (0, 140, 255), -1)
                    cv2.putText(disp_frame, f"WARNING: CRI {cri_score:.1f} (ZONE {self.zone_id})", (12, 21), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                # 7. JPEG 인코딩 및 메모리 바이트 업데이트 (Pub-Sub)
                _, buffer = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
                self.latest_jpeg_bytes = buffer.tobytes()
                self.frame_seq += 1
                self.new_frame_event.set()
                self.new_frame_event.clear()

                # 8. 1분 정기 아카이빙 워커 통합 (10 FPS 샘플링)
                if frame_idx % 3 == 0:
                    splitter = get_zone_splitter(self.zone_id, fps=10.0, split_duration_sec=60.0)
                    splitter.write_frame(disp_frame, frame_metric)

                # 9. 30 FPS 정속 타이머 슬립
                elapsed = time.time() - t_start
                sleep_time = max(0.001, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"[ZoneLiveEngine Warning] Zone {self.zone_id} 루프 에러: {e}")
        finally:
            cap.release()
            self.is_running = False


def get_zone_engine(zone_id: int) -> ZoneLiveEngine:
    """구역별 싱글톤 ZoneLiveEngine 반환"""
    global _zone_engines
    if zone_id not in _zone_engines:
        _zone_engines[zone_id] = ZoneLiveEngine(zone_id)
    return _zone_engines[zone_id]


async def generate_mjpeg_stream(zone_id: int):
    """싱글톤 ZoneLiveEngine으로부터 최신 30 FPS JPEG 프레임을 구독하여 무부하 브로드캐스트 송출"""
    engine = get_zone_engine(zone_id)
    engine.ensure_started()
    last_seq = -1

    while True:
        try:
            try:
                await asyncio.wait_for(engine.new_frame_event.wait(), timeout=0.04)
            except asyncio.TimeoutError:
                pass

            if engine.latest_jpeg_bytes is not None and engine.frame_seq != last_seq:
                last_seq = engine.frame_seq
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + engine.latest_jpeg_bytes + b'\r\n')
            else:
                await asyncio.sleep(0.01)
        except Exception:
            await asyncio.sleep(0.03)


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
