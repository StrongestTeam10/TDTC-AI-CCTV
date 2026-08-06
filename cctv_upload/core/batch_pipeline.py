# realtime_cctv_batch_pipeline.py
# CSRNet 기반 3개 구역 실시간 영상 처리 파이프라인 (Real-Time Batch Pipeline Engine)

import os
import sys
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from core.shared_engine import SharedCSRNetEngine
except ImportError:
    from shared_engine import SharedCSRNetEngine

# =========================================================================
# 1. 설정 및 경로
# =========================================================================
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
CCTV_UPLOAD_DIR = os.path.dirname(CORE_DIR)
AI_PIPELINE_DIR = os.path.dirname(CCTV_UPLOAD_DIR)
WORKSPACE_DIR = os.path.dirname(AI_PIPELINE_DIR)

RESULTS_DIR = os.path.join(AI_PIPELINE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("CSRNET_MODEL_PATH")
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(AI_PIPELINE_DIR, "models", "csrnet_ultimate_epoch_8.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CORE_DIR, "models", "csrnet_ultimate_epoch_8.pth")

TARGET_FPS = float(os.environ.get("TARGET_FPS", "10.0"))


# 이미지 정규화 변환 (CSRNet / VGG16)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 호모그래피 캘리브레이션 행렬
H_MATRIX = np.array([
    [ 0.015, -0.002, -8.50],
    [ 0.001,  0.022, -2.10],
    [ 0.000,  0.001,  1.00]
])

def transform_pixel_to_bev(u: float, v: float, H: np.ndarray = H_MATRIX) -> Tuple[float, float]:
    """2D 픽셀 좌표 (u, v) -> 라이다 BEV (X, Y) 미터 좌표 변환"""
    pt = np.array([u, v, 1.0]).reshape(3, 1)
    bev_pt = np.dot(H, pt)
    bev_pt /= bev_pt[2]
    return float(bev_pt[0][0]), float(bev_pt[1][0])

# =========================================================================
# 2. Dataclass 및 Thread-safe Queue
# =========================================================================
@dataclass
class FramePackage:
    zone_id: int              # 구역 번호 (1, 2, 3)
    frame_idx: int            # 프레임 번호
    timestamp: float          # 타임스탬프 (초)
    tensor: torch.Tensor      # CPU float32/float16 이미지 텐서 (3, 720, 1280)
    is_dummy: bool = False    # 패딩용 더미 텐서 여부
    is_eof: bool = False      # 영상 종료 패킷 여부

# =========================================================================
# 3. GPU MaxPool2d 기반 Peak 좌표 추출 (Zero-Resize)
# =========================================================================
def extract_peaks_gpu(density_tensor: torch.Tensor, threshold: float = 0.01) -> List[Tuple[float, float]]:
    """
    GPU 텐서 (1, 1, 90, 160) 상태에서 MaxPool2d로 0.1ms 내 NMS 극대점 및 720p Scale-back 복원
    """
    device = density_tensor.device
    h_out, w_out = density_tensor.shape[-2], density_tensor.shape[-1] # 90, 160
    
    # 1. GPU 5x5 MaxPool2d Peak 추출
    max_pooled = F.max_pool2d(density_tensor, kernel_size=5, stride=1, padding=2)
    peaks_mask = (density_tensor == max_pooled) & (density_tensor > threshold)
    
    # 2. Peak 픽셀 인덱스 찾기
    peak_indices = torch.nonzero(peaks_mask.squeeze(), as_tuple=False) # Shape: (N, 2) -> (y, x)
    if peak_indices.numel() == 0:
        return []
    
    # 3. 720p 스케일 복원 곱연산
    scale_x = 1280.0 / float(w_out) # 8.0
    scale_y = 720.0 / float(h_out)  # 8.0
    
    peaks = []
    for idx in peak_indices:
        py_raw = float(idx[0].item())
        px_raw = float(idx[1].item())
        px_720 = px_raw * scale_x
        py_720 = py_raw * scale_y
        peaks.append((px_720, py_720))
        
    return peaks

# =========================================================================
# 4. 실시간 배치 파이프라인 메인 클래스
# =========================================================================
class RealtimeBatchPipeline:
    def __init__(self, zone_video_map: Dict[int, str], target_fps: float = 10.0):
        self.zone_video_map = zone_video_map
        self.target_fps = target_fps
        self.frame_queue = queue.Queue(maxsize=30)
        self.results_lock = threading.Lock()
        self.zone_results = {1: [], 2: [], 3: []}
        
        # Single Dedicated GPU Engine 로드
        self.engine = SharedCSRNetEngine(MODEL_PATH)
        self.stop_event = threading.Event()

    def _video_reader_thread(self, zone_id: int, video_path: str):
        """구역별 비디오 읽기 및 10 FPS 스킵, Queue Push 스레드"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Zone {zone_id} 비디오 오픈 실패: {video_path}")
            self.frame_queue.put(FramePackage(zone_id=zone_id, frame_idx=-1, timestamp=0.0, tensor=torch.zeros(3, 720, 1280), is_eof=True))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip_interval = max(1, int(round(fps / self.target_fps)))
        frame_count = 0

        while cap.isOpened() and not self.stop_event.is_set():
            frame_count += 1
            if (frame_count - 1) % skip_interval != 0:
                if not cap.grab(): # cap.grab() 포인터 스킵 최적화
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # 720p 가공 및 전처리
            if frame.shape[:2] != (720, 1280):
                frame = cv2.resize(frame, (1280, 720))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = img_transform(rgb_frame) # (3, 720, 1280)
            timestamp = (frame_count - 1) / fps

            pkg = FramePackage(
                zone_id=zone_id,
                frame_idx=frame_count,
                timestamp=timestamp,
                tensor=tensor_frame
            )
            self.frame_queue.put(pkg)

        cap.release()
        # EOF 알림 패킷
        self.frame_queue.put(FramePackage(zone_id=zone_id, frame_idx=frame_count, timestamp=0.0, tensor=torch.zeros(3, 720, 1280), is_eof=True))
        print(f"[READER DONE] Zone {zone_id} 읽기 완료 (총 {frame_count} 프레임)")

    def _gpu_worker_thread(self, active_zones_count: int):
        """단일 dedicated GPU Worker 스레드 (Dynamic Timeout + Dummy Padding + 실시간 진행률 출력)"""
        eof_zones = set()
        zero_dummy = torch.zeros(3, 720, 1280)
        processed_frames = 0
        total_estimate_frames = 1824 # 3구역 각 1분 10fps 기준 약 1800프레임
        last_log_time = time.time()

        print("[GPU WORKER] Dedicated GPU Worker 스레드 가동 시작...", flush=True)
        
        while len(eof_zones) < active_zones_count and not self.stop_event.is_set():
            batch_pkgs: List[FramePackage] = []
            start_wait = time.time()
            
            # 30ms Dynamic Timeout Batching 수집
            while len(batch_pkgs) < 3 and (time.time() - start_wait) < 0.030:
                try:
                    pkg = self.frame_queue.get(timeout=0.005)
                    if pkg.is_eof:
                        eof_zones.add(pkg.zone_id)
                    else:
                        batch_pkgs.append(pkg)
                except queue.Empty:
                    pass

            if not batch_pkgs:
                continue

            # Dummy Zero-Tensor Padding (batch_size=3 고정 유지)
            real_len = len(batch_pkgs)
            while len(batch_pkgs) < 3:
                dummy_pkg = FramePackage(zone_id=-1, frame_idx=-1, timestamp=0.0, tensor=zero_dummy, is_dummy=True)
                batch_pkgs.append(dummy_pkg)

            # (3, 3, 720, 1280) Mini-Batch Tensor 구성
            batch_tensors = torch.stack([p.tensor for p in batch_pkgs], dim=0)

            # GPU Shared Model Batch 추론 (1회 실행)
            t0 = time.time()
            with torch.no_grad():
                density_outputs = self.engine.infer_batch(batch_tensors) # (3, 1, 90, 160)
            infer_dt = time.time() - t0

            # Unbatching & Peak 추출
            for idx in range(real_len):
                pkg = batch_pkgs[idx]
                if pkg.is_dummy or pkg.is_eof:
                    continue

                processed_frames += 1
                density_single = density_outputs[idx:idx+1] # (1, 1, 90, 160)
                peaks_720 = extract_peaks_gpu(density_single)
                
                # BEV 물리 좌표 매핑
                bev_coords = []
                for px, py in peaks_720:
                    bx, by = transform_pixel_to_bev(px, py)
                    bev_coords.append({"px": px, "py": py, "bev_x": bx, "bev_y": by})

                record = {
                    "zone_id": pkg.zone_id,
                    "frame_idx": pkg.frame_idx,
                    "timestamp": round(pkg.timestamp, 3),
                    "pedestrian_count": len(peaks_720),
                    "coordinates": bev_coords,
                    "infer_dt_ms": round(infer_dt * 1000 / real_len, 2)
                }

                with self.results_lock:
                    self.zone_results[pkg.zone_id].append(record)

            # 주기적 진행률 로그 출력 (매 1초 또는 30프레임마다)
            if time.time() - last_log_time >= 1.0 or processed_frames % 30 == 0:
                pct = min(100.0, (processed_frames / total_estimate_frames) * 100) if total_estimate_frames > 0 else 0
                current_fps = round(real_len / max(0.001, infer_dt), 1)
                print(f"[PROGRESS] Frame: {processed_frames}/{total_estimate_frames} ({pct:.1f}%) | GPU Speed: {current_fps} FPS", flush=True)
                last_log_time = time.time()


        print("[GPU WORKER] Dedicated GPU Worker 스레드 종료.", flush=True)


    def run(self) -> Dict[int, List[Dict]]:
        """파이프라인 실행 유틸리티"""
        start_time = time.time()
        print("=" * 80, flush=True)
        print(" [CSRNet 3개 구역 실시간 통합 배치 파이프라인 가동] ", flush=True)
        print(f" Target FPS: {self.target_fps} | GPU Worker: Single Dedicated | Batch Size: 3", flush=True)
        print("=" * 80, flush=True)

        reader_threads = []
        for zone_id, video_path in self.zone_video_map.items():
            if os.path.exists(video_path):
                t = threading.Thread(target=self._video_reader_thread, args=(zone_id, video_path))
                t.start()
                reader_threads.append(t)

        gpu_thread = threading.Thread(target=self._gpu_worker_thread, args=(len(reader_threads),))
        gpu_thread.start()

        for t in reader_threads:
            t.join()
        gpu_thread.join()

        elapsed = time.time() - start_time
        total_records = sum(len(r) for r in self.zone_results.values())
        print("=" * 80, flush=True)
        print(f"[SUCCESS] 파이프라인 완주 완료! 총 소요 시간: {elapsed:.2f}초 | 수집 데이터: {total_records}건", flush=True)
        print("=" * 80, flush=True)

        return self.zone_results



# =========================================================================
# 5. 실행 및 테스트
# =========================================================================
if __name__ == "__main__":
    TEST_DIR = os.path.join(AI_PIPELINE_DIR, "test")
    if not os.path.exists(TEST_DIR):
        TEST_DIR = os.path.join(WORKSPACE_DIR, "test")
    
    # test/zone_id X 비디오 스캔
    zone_map = {}
    for z in [1, 2, 3]:
        z_dir = os.path.join(TEST_DIR, f"zone_id {z}")
        if os.path.exists(z_dir):
            files = [os.path.join(z_dir, f) for f in os.listdir(z_dir) if f.endswith(".mp4")]
            if files:
                zone_map[z] = files[0] # 첫 번째 비디오 테스트


    if not zone_map:
        print("[ERROR] 테스트 비디오를 찾을 수 없습니다. (test/zone_id X/ .mp4 확인)")
        sys.exit(1)

    pipeline = RealtimeBatchPipeline(zone_map, target_fps=TARGET_FPS)
    results = pipeline.run()
