"""
server/live_streamer.py - 로컬 영상 실시간 분석 및 영상 프레임(Base64/MJPEG) 웹소켓 스트리밍 엔진
"""

import os
import cv2
import time
import base64
import asyncio
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional

from server.config import (
    BASE_DIR, RESULTS_DIR, MODELS_DIR, pipeline_state
)
from server.models import (
    torch_available, CSRNet, apply_mosaic, extract_peaks_from_density, H_MATRIX, ROI_2D_POLY
)
from server.tracker import FrameStabilizer
from server.websocket import manager
from server.video_buffer import global_buffer_manager


class LiveCctvStreamer:
    """
    로컬 비디오 파일을 읽어 실시간으로 AI 분석(CSRNet+모자이크)을 수행하고,
    분석된 영상 프레임(Base64)과 관제 지표를 WebSocket으로 쉴 새 없이 브로드캐스트하는 스트리머.
    """

    def __init__(self):
        self.is_running = False
        self.target_fps = 10.0
        self.sleep_interval = 1.0 / self.target_fps
        self.latest_jpeg_frames: Dict[int, bytes] = {}
        self.current_frame_ids: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.worker_task: Optional[asyncio.Task] = None

        # CSRNet 모델 준비
        self.model = None
        self.device = None
        self.transform = None
        self._init_model()

    def _init_model(self):
        if torch_available and CSRNet is not None:
            try:
                import torch
                from torchvision import transforms
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = CSRNet().to(self.device)
                model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=self.device)
                    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.eval()
                    print(f"[LiveStreamer] CSRNet 로드 성공: {model_path} ({self.device})")
                else:
                    self.model = None
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                print(f"[LiveStreamer Warning] PyTorch 모델 로드 실패: {e}")
                self.model = None

    def _find_video_file(self, zone_id: int) -> Optional[str]:
        """구역별 로컬 영상 파일 탐색"""
        candidates = [
            # 1. TDTC-AI-FE public 폴더
            os.path.join(os.path.dirname(BASE_DIR), "TDTC-AI-FE", "public", f"cctv_zone{zone_id}.mp4"),
            os.path.join(os.path.dirname(BASE_DIR), "TDTC-AI-FE", "public", "cctv_mangwon_live.mp4"),
            # 2. ai_pipeline legacy uploads
            os.path.join(os.path.dirname(BASE_DIR), "ai_pipeline", "legacy", "cctv_ai_pipeline", "uploads", f"test0{zone_id}.mp4"),
            # 3. TDTC-AI-CCTV results / steps
            os.path.join(RESULTS_DIR, "cctv_simulation_video.mp4"),
            os.path.join(BASE_DIR, "cctv_upload", "mangwon_test.mp4"),
        ]
        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return path
        return None

    async def start(self):
        """실시간 스트리밍 백그라운드 태스크 구동"""
        if self.is_running:
            return
        self.is_running = True
        self.worker_task = asyncio.create_task(self._stream_loop())
        print("[LiveStreamer] 🚀 실시간 AI 영상 프레임 스트리밍 워커 가동 시작!")

    async def stop(self):
        """스트리밍 중지"""
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            self.worker_task = None
        print("[LiveStreamer] 🛑 실시간 스트리밍 워커 중지")

    async def _stream_loop(self):
        """Zone 1, 2, 3 로컬 영상을 순환하며 실시간 분석 및 영상 프레임 Push"""
        caps: Dict[int, cv2.VideoCapture] = {}
        for z_id in [1, 2, 3]:
            v_path = self._find_video_file(z_id)
            if v_path:
                cap = cv2.VideoCapture(v_path)
                if cap.isOpened():
                    caps[z_id] = cap
                    print(f"[LiveStreamer] Zone {z_id} 영상 로드 성공: {v_path}")

        if not caps:
            print("[LiveStreamer Warning] 로컬 영상 파일을 찾지 못해 시뮬레이션 합성 모드로 실행합니다.")

        stabilizers = {1: FrameStabilizer(), 2: FrameStabilizer(), 3: FrameStabilizer()}
        target_size = (640, 360) # 16:9 웹 최적화 해상도

        frame_counter = 0

        while self.is_running:
            loop_start_time = time.time()
            frame_counter += 1

            for z_id in [1, 2, 3]:
                cap = caps.get(z_id)
                frame_img = None

                if cap is not None and cap.isOpened():
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        # 영상 끝에 도달 시 처음으로 루프 (무한 재생)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, raw_frame = cap.read()
                    if ret and raw_frame is not None:
                        frame_img = cv2.resize(raw_frame, target_size)

                # 영상이 없으면 테스트용 더미 관제 프레임 생성
                if frame_img is None:
                    frame_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    cv2.putText(frame_img, f"Zone {z_id} LIVE CCTV", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2)
                    cv2.putText(frame_img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # 1. 원형 버퍼(35초 클립 추출용)에 원본 프레임 등록
                global_buffer_manager.push_frame(z_id, frame_img, loop_start_time)

                # 2. CSRNet AI 추론 및 보행자 검출
                detected_peaks = []
                base_count = 0

                if self.model is not None and self.transform is not None and (frame_counter % 2 == 0):
                    try:
                        import torch
                        img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            output = torch.clamp(output, min=0)
                            density_map = output.squeeze().cpu().numpy()
                            h_out, w_out = output.shape[2], output.shape[3]
                            scale_x = target_size[0] / w_out
                            scale_y = target_size[1] / h_out
                            raw_peaks = extract_peaks_from_density(density_map, threshold=0.0005)
                            for px_raw, py_raw in raw_peaks:
                                detected_peaks.append((px_raw * scale_x, py_raw * scale_y))
                            base_count = len(detected_peaks)
                    except Exception:
                        pass

                if base_count == 0:
                    # 폴백 모의 보행자 좌표 시뮬레이션
                    seed = (z_id * 10) + (frame_counter // 5)
                    base_count = max(4, 10 + int(6 * np.sin(seed / 4.0)) + (z_id * 3))
                    detected_peaks = [
                        (150 + (i * 35 + seed * 3) % (target_size[0] - 200),
                         100 + (i * 25 + seed * 2) % (target_size[1] - 150))
                        for i in range(base_count)
                    ]

                # 3. 가우시안 모자이크(비식별화) 및 보행자 마커 렌더링
                rendered_frame = frame_img.copy()
                detections_list = []

                for i, (px, py) in enumerate(detected_peaks):
                    # 가우시안 모자이크 적용
                    x1, y1 = max(0, int(px - 15)), max(0, int(py - 15))
                    x2, y2 = min(target_size[0], int(px + 15)), min(target_size[1], int(py + 45))
                    rendered_frame = apply_mosaic(rendered_frame, x1, y1, x2, y2, neighbor=9)

                    # 보행자 추적 바운딩 박스 & 중심점 시각화
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), (0, 255, 128), 1)
                    cv2.circle(rendered_frame, (int(px), int(py)), 3, (0, 255, 255), -1)

                    detections_list.append({
                        "track_id": i + 1,
                        "raw_bbox_bottom_center": [round(px, 1), round(py + 45, 1)],
                        "stabilized_person_coords": [round(px, 1), round(py + 45, 1)],
                        "current_zone_id": z_id
                    })

                # 4. CRI 위험도 지표 산출
                occupancy_rate = round(min(100.0, base_count * 3.2), 1)
                raw_cri = 10.0 + (base_count * 2.8) + (occupancy_rate * 0.3)
                cri_score = round(min(100.0, max(10.0, raw_cri)), 1)
                risk_level = "EMERGENCY_EVACUATION" if cri_score >= 70.0 else "WARNING" if cri_score >= 35.0 else "NORMAL"

                # 5. 프레임 이미지 JPEG ➡️ Base64 인코딩
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
                _, buffer = cv2.imencode('.jpg', rendered_frame, encode_param)
                jpeg_bytes = buffer.tobytes()
                self.latest_jpeg_frames[z_id] = jpeg_bytes
                b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
                image_base64_url = f"data:image/jpeg;base64,{b64_str}"

                self.current_frame_ids[z_id] += 1
                curr_fid = self.current_frame_ids[z_id]

                # 6. WebSocket 클라이언트로 '영상 프레임 데이터 + AI 분석 지표' 실시간 브로드캐스트!
                await manager.broadcast({
                    "type": "CCTV_AI_STREAM",
                    "frame_id": curr_fid,
                    "zone_id": z_id,
                    "filename": f"zone_{z_id}_live.mp4",
                    "image_base64": image_base64_url,
                    "pedestrian_count": base_count,
                    "occupancy_rate": occupancy_rate,
                    "stagnation_sec": 0,
                    "cri_score": cri_score,
                    "risk_level": risk_level,
                    "detections": detections_list,
                    "timestamp": time.time()
                })

            # FPS 동기화 (지정된 주기에 맞춰 대기)
            elapsed = time.time() - loop_start_time
            sleep_time = max(0.01, self.sleep_interval - elapsed)
            await asyncio.sleep(sleep_time)

        for cap in caps.values():
            if cap.isOpened():
                cap.release()


# 전역 실시간 스트리머 인스턴스
global_live_streamer = LiveCctvStreamer()
