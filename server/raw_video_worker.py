"""
server/raw_video_worker.py - 1분 단위 상시 원본 영상(Raw Video) 스플릿 및 S3/DB(vdoclip01m, pedaggr01h, mrkrisk01m) 자동 적재 워커
"""

import os
import cv2
import json
import time
import requests
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from server.config import (
    RESULTS_DIR, BACKEND_URL, BACKEND_API_KEY
)
from server.s3_uploader import upload_file_to_s3


class RawVideoSplitter:
    """
    스트림 프레임을 받아 1분(60초) 단위로 영상을 분할 저장하고,
    S3의 raw-videos/ 폴더에 업로드 후
    Supabase/RDS DB (vdoclip01m, pedaggr01h, mrkrisk01m)에 100% 무인 직접 적재합니다.
    """

    def __init__(self, zone_id: int = 1, fps: float = 10.0, split_duration_sec: float = 60.0):
        self.zone_id = zone_id
        self.fps = fps
        self.split_duration_sec = split_duration_sec
        self.max_frames_per_split = int(fps * split_duration_sec)

        self.current_writer = None
        self.current_file_path = None
        self.current_start_time = None
        self.frame_count = 0
        self.target_size = None
        self.buffered_metrics: List[dict] = []
        self._lock = threading.Lock()

    def _open_new_segment(self, frame_sample):
        """새 1분 단위 파일 Writer 초기화 (10 FPS)"""
        h, w = frame_sample.shape[:2]
        self.target_size = (w, h)
        now_dt = datetime.now(timezone.utc)
        self.current_start_time = now_dt
        
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"raw_zone_{self.zone_id}_{timestamp_str}.mp4"
        self.current_file_path = os.path.join(RESULTS_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_writer = cv2.VideoWriter(self.current_file_path, fourcc, self.fps, self.target_size)
        self.frame_count = 0
        self.buffered_metrics = []
        print(f"[RawVideoSplitter] Zone {self.zone_id} 새 1분 세그먼트 생성 시작: {self.current_file_path}")

    def _close_and_upload_segment(self):
        """현재 세그먼트 완료 및 S3 업로드 + 백엔드 Webhook 적재"""
        if self.current_writer is not None:
            self.current_writer.release()
            self.current_writer = None

        if self.current_file_path and os.path.exists(self.current_file_path):
            file_to_upload = self.current_file_path
            start_time_iso = self.current_start_time.isoformat().replace("+00:00", "Z")
            end_time_dt = self.current_start_time + timedelta(seconds=(self.frame_count / self.fps))
            end_time_iso = end_time_dt.isoformat().replace("+00:00", "Z")
            filename = os.path.basename(file_to_upload)
            metrics_snapshot = list(self.buffered_metrics)

            def _async_upload_and_db_insert():
                # 1. S3 업로드 (raw-videos/ 폴더)
                s3_key = f"raw-videos/{filename}"
                s3_url = upload_file_to_s3(file_to_upload, s3_key)
                print(f"[RawVideoSplitter] Zone {self.zone_id} 1분 영상 S3 업로드 완료: {s3_url}")

                # 2. 트랙 1 웹훅으로 클립 등록 -> clipId 획득 (직접 INSERT 대체)
                created_clip_id = None
                try:
                    r = requests.post(
                        f"{BACKEND_URL.rstrip('/')}/api/v1/video-clips",
                        headers={"X-API-KEY": BACKEND_API_KEY},
                        json={
                            "zoneId": self.zone_id,
                            "clipType": "TEMP",
                            "s3ClipUrl": s3_url,
                            "startTime": start_time_iso,
                            "endTime": end_time_iso,
                        },
                        timeout=15,
                    )
                    r.raise_for_status()
                    created_clip_id = r.json().get("clipId")
                    print(f"[RawVideoSplitter] ✅ Zone {self.zone_id} 클립 웹훅 등록 성공: clipId = {created_clip_id}")
                except Exception as e:
                    print(f"[RawVideoSplitter Warning] 클립 웹훅 실패: {e}")

                # 3. 트랙 2 벌크 웹훅으로 1분치 프레임 적재 (다이렉트 INSERT 대체)
                if metrics_snapshot and created_clip_id:
                    frames = [{
                        "frameId": m.get("frame_id"),
                        "videoId": m.get("video_id", 1),
                        "pixelsJson": m.get("pixels_json", "{}"),
                        "bevXyzJson": m.get("bev_xyz_json", "{}"),
                        "capturedAt": m.get("captured_at"),
                        "totalCount": m.get("total_count", 0),
                        "riskScore": m.get("cri_score", 0.0),
                        "riskLevel": m.get("risk_level", "NORMAL"),
                        "reasonCode": "AI_REALTIME_CRI",
                        "occupancyRate": m.get("occupancy_rate", 0.0),
                        "stagnationSec": m.get("stagnation_sec", 0.0),
                        "videoUrl": s3_url,
                    } for m in metrics_snapshot]

                    try:
                        r = requests.post(
                            f"{BACKEND_URL.rstrip('/')}/api/v1/metrics/bulk",
                            headers={"X-API-KEY": BACKEND_API_KEY},
                            json={
                                "clipId": created_clip_id,
                                "zoneId": self.zone_id,
                                "frames": frames
                            },
                            timeout=30,
                        )
                        if r.status_code == 409:
                            print(f"[RawVideoSplitter] Zone {self.zone_id} 이미 적재된 분(409) - 건너뜀")
                        else:
                            r.raise_for_status()
                            res_json = r.json()
                            c_cnt = res_json.get("insertedCoordinates", len(frames))
                            r_cnt = res_json.get("insertedRisks", len(frames))
                            print(f"[RawVideoSplitter] ✅ Zone {self.zone_id} 1분 메트릭 벌크 적재 완료: clipId={created_clip_id} | pedaggr01h(좌표)={c_cnt}건 | mrkrisk01m(위험도)={r_cnt}건")
                    except Exception as e:
                        print(f"[RawVideoSplitter Warning] 벌크 웹훅 실패: {e}")

                # 4. 로컬 임시 파일 정리
                try:
                    if os.path.exists(file_to_upload):
                        os.remove(file_to_upload)
                except Exception:
                    pass

            # 백그라운드 스레드로 비동기 처리하여 스트리밍 프레임 드랍 방지
            t = threading.Thread(target=_async_upload_and_db_insert, daemon=True)
            t.start()

        self.current_file_path = None
        self.frame_count = 0
        self.buffered_metrics = []

    def write_frame(self, frame, metrics: Optional[dict] = None):
        """매 프레임 기록 및 1분 도달 시 자동 분할/S3/DB 적재"""
        with self._lock:
            if self.current_writer is None:
                self._open_new_segment(frame)

            if self.target_size and frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(frame, self.target_size)

            self.current_writer.write(frame)
            self.frame_count += 1

            if metrics:
                self.buffered_metrics.append(metrics)

            # 녹화 진행 상황 로그 출력 (5초마다)
            if self.frame_count % int(self.fps * 5) == 0:
                print(f"[RawVideoSplitter] Zone {self.zone_id} 녹화 진행 중: {self.frame_count}/{self.max_frames_per_split} frames ({self.frame_count/self.fps:.0f}s / {self.split_duration_sec:.0f}s)")

            # 최대 분할 시간 도달 시 분할 및 S3 + DB 적재
            if self.frame_count >= self.max_frames_per_split:
                print(f"[RawVideoSplitter] Zone {self.zone_id} 분할 기준 도달 ({self.frame_count} frames) -> S3 및 DB 적재 트리거!")
                self._close_and_upload_segment()

    def flush_segment(self):
        """원본 영상 1회 완주 시 즉시 현재 세그먼트 마감 및 S3/DB 적재"""
        with self._lock:
            if self.frame_count >= int(self.fps * 3): # 최소 3초 이상 녹화되었을 때 마감
                print(f"[RawVideoSplitter] Zone {self.zone_id} 원본 영상 1회 완주 ({self.frame_count} frames, {self.frame_count/self.fps:.1f}s) -> 즉시 S3 및 DB 적재 트리거!")
                self._close_and_upload_segment()

    def finalize(self):
        """스트림 종료 시 남아있는 버퍼 파일 업로드"""
        with self._lock:
            if self.frame_count > 0:
                self._close_and_upload_segment()


def _upload_and_send_webhooks(zone_id: int, file_to_upload: str, start_time_dt: datetime, frame_count: int, fps: float, metrics_snapshot: list):
    """S3 업로드 및 백엔드 Webhook 적재 공통 헬퍼"""
    start_time_iso = start_time_dt.isoformat().replace("+00:00", "Z")
    end_time_dt = start_time_dt + timedelta(seconds=(frame_count / fps))
    end_time_iso = end_time_dt.isoformat().replace("+00:00", "Z")
    filename = os.path.basename(file_to_upload)

    def _async_task():
        # 1. S3 업로드 (raw-videos/ 폴더)
        s3_key = f"raw-videos/{filename}"
        s3_url = upload_file_to_s3(file_to_upload, s3_key)
        print(f"[1Min Clip Renderer] ✅ Zone {zone_id} 1분 완주 모자이크 영상 S3 업로드 완료: {s3_url}")

        # 2. 트랙 1 웹훅: 클립 등록 -> clipId 획득
        created_clip_id = None
        try:
            r = requests.post(
                f"{BACKEND_URL.rstrip('/')}/api/v1/video-clips",
                headers={"X-API-KEY": BACKEND_API_KEY},
                json={
                    "zoneId": zone_id,
                    "clipType": "TEMP",
                    "s3ClipUrl": s3_url,
                    "startTime": start_time_iso,
                    "endTime": end_time_iso,
                },
                timeout=15,
            )
            r.raise_for_status()
            created_clip_id = r.json().get("clipId")
            print(f"[1Min Clip Renderer] ✅ Zone {zone_id} 클립 웹훅 등록 성공: clipId = {created_clip_id}")
        except Exception as e:
            print(f"[1Min Clip Renderer Warning] 클립 웹훅 실패: {e}")

        # 3. 트랙 2 웹훅: 600프레임 메트릭 벌크 적재
        if metrics_snapshot and created_clip_id:
            frames = [{
                "frameId": m.get("frame_id"),
                "videoId": m.get("video_id", 1),
                "pixelsJson": m.get("pixels_json", "{}"),
                "bevXyzJson": m.get("bev_xyz_json", "{}"),
                "capturedAt": m.get("captured_at"),
                "totalCount": m.get("total_count", 0),
                "riskScore": m.get("cri_score", 0.0),
                "riskLevel": m.get("risk_level", "NORMAL"),
                "reasonCode": "AI_REALTIME_CRI",
                "occupancyRate": m.get("occupancy_rate", 0.0),
                "stagnationSec": m.get("stagnation_sec", 0.0),
                "videoUrl": s3_url,
            } for m in metrics_snapshot]

            try:
                r = requests.post(
                    f"{BACKEND_URL.rstrip('/')}/api/v1/metrics/bulk",
                    headers={"X-API-KEY": BACKEND_API_KEY},
                    json={
                        "clipId": created_clip_id,
                        "zoneId": zone_id,
                        "frames": frames
                    },
                    timeout=30,
                )
                if r.status_code == 409:
                    print(f"[1Min Clip Renderer] Zone {zone_id} 이미 적재된 분(409) - 건너뜀")
                else:
                    r.raise_for_status()
                    res_json = r.json()
                    c_cnt = res_json.get("insertedCoordinates", len(frames))
                    r_cnt = res_json.get("insertedRisks", len(frames))
                    print(f"[1Min Clip Renderer] ✅ Zone {zone_id} 1분(600프레임) 벌크 적재 완료: clipId={created_clip_id} | pedaggr01h(보행자좌표)={c_cnt}건 | mrkrisk01m(위험도)={r_cnt}건")
            except Exception as e:
                print(f"[1Min Clip Renderer Warning] 벌크 웹훅 실패: {e}")

        # 4. 로컬 임시 파일 정리
        try:
            if os.path.exists(file_to_upload):
                os.remove(file_to_upload)
        except Exception:
            pass

    t = threading.Thread(target=_async_task, daemon=True)
    t.start()


def generate_and_upload_zone_1min_clip(zone_id: int, video_path: str, yolo_model=None, target_fps: float = 10.0, target_duration_sec: float = 60.0):
    """
    원본 영상에서 정확히 1회 완주(1분 분량)를 10 FPS(600장)로 균등 추출하고
    보행자 모자이크를 입혀 S3 및 백엔드 Webhook에 적재하는 독립 백그라운드 렌더러
    """
    from server.models import apply_mosaic

    if not os.path.exists(video_path):
        print(f"[1Min Clip Renderer] 비디오 파일 없음: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 10 FPS로 60초(600장)를 원본 전체에서 균등하게 추출하기 위한 Stride 계산
    stride = max(1, int(round(orig_fps / target_fps)))

    now_dt = datetime.now(timezone.utc)
    timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
    out_filename = f"raw_zone_{zone_id}_{timestamp_str}.mp4"
    out_path = os.path.join(RESULTS_DIR, out_filename)

    writer = None
    frame_idx = 0
    written_frames = 0
    buffered_metrics = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 원본 1회 끝남 (되감기 0회!)

            frame_idx += 1
            if frame_idx % stride != 0:
                continue  # Stride에 맞지 않는 프레임은 건너뜀

            # 좌우 검은색 필러박스 크롭 (가로 영상 꽉 차게)
            h, w = frame.shape[:2]
            if w > h and w >= 1900 and h >= 1000:
                frame = frame[:, 656:1264]

            # Writer 초기화
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_path, fourcc, target_fps, (fw, fh))

            # YOLO 보행자 모자이크 적용
            detected_boxes = []
            if yolo_model is not None:
                try:
                    res = yolo_model(frame, classes=[0], verbose=False, conf=0.25, imgsz=384)
                    if len(res) > 0 and len(res[0].boxes) > 0:
                        for b in res[0].boxes:
                            coords = b.xyxy[0].cpu().numpy()
                            detected_boxes.append((float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])))
                except Exception:
                    pass

            if detected_boxes:
                for (x1, y1, x2, y2) in detected_boxes:
                    frame = apply_mosaic(frame, float(x1), float(y1), float(x2), float(y2))

            writer.write(frame)
            written_frames += 1

            # 메트릭 패킹
            pixels_dict = {f"p_{i+1}": [int((x1+x2)/2), int(y2)] for i, (x1, y1, x2, y2) in enumerate(detected_boxes)}
            bev_dict = {f"p_{i+1}": [round(float((x1+x2)/2 * 0.015), 2), round(float(y2 * 0.02), 2), 0.0] for i, (x1, y1, x2, y2) in enumerate(detected_boxes)}
            buffered_metrics.append({
                "zone_id": zone_id,
                "frame_id": written_frames,
                "video_id": 1,
                "total_count": len(detected_boxes),
                "occupancy_rate": min(100.0, round(len(detected_boxes) * 2.2, 1)),
                "stagnation_sec": 0.0,
                "cri_score": min(100.0, max(5.0, round(len(detected_boxes) * 1.8, 1))),
                "risk_level": "SAFE" if len(detected_boxes) < 20 else "WARNING",
                "pixels_json": json.dumps(pixels_dict),
                "bev_xyz_json": json.dumps(bev_dict),
                "captured_at": (now_dt + timedelta(seconds=(written_frames / target_fps))).isoformat().replace("+00:00", "Z")
            })

            # 최대 600프레임(약 60초) 도달 시 종료
            if written_frames >= int(target_fps * target_duration_sec):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # S3 업로드 및 백엔드 Webhook 적재 실행
    if written_frames > 0 and os.path.exists(out_path):
        print(f"[1Min Clip Renderer] Zone {zone_id} 1회 완주 렌더링 완료 ({written_frames} frames, {written_frames/target_fps:.1f}s) -> S3/DB 적재 시작")
        _upload_and_send_webhooks(zone_id, out_path, now_dt, written_frames, target_fps, buffered_metrics)
        return out_path
    return None


_zone_splitters = {}

def get_zone_splitter(zone_id: int, fps: float = 10.0, split_duration_sec: float = 60.0) -> RawVideoSplitter:
    """구역별 RawVideoSplitter 싱글톤 반환 (기본: 10 FPS, 60초 분할)"""
    global _zone_splitters
    if zone_id not in _zone_splitters:
        _zone_splitters[zone_id] = RawVideoSplitter(zone_id=zone_id, fps=fps, split_duration_sec=split_duration_sec)
    return _zone_splitters[zone_id]
