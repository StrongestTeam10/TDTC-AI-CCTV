"""
server/raw_video_worker.py - 1분 단위 상시 원본 영상(Raw Video) 스플릿 및 S3/DB 적재 워커
"""

import os
import cv2
import time
import requests
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

from server.config import (
    RESULTS_DIR, BACKEND_URL, BACKEND_API_KEY
)
from server.s3_uploader import upload_file_to_s3


class RawVideoSplitter:
    """
    스트림 프레임을 받아 1분(60초) 단위로 영상을 분할 저장하고,
    S3의 raw-videos/ 폴더에 업로드 후 Java 백엔드(/api/v1/video-clips)에 등록합니다.
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
        self._lock = threading.Lock()

    def _open_new_segment(self, frame_sample):
        """새 1분 단위 파일 Writer 초기화"""
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
        print(f"[RawVideoSplitter] Zone {self.zone_id} 새 1분 세그먼트 생성 시작: {self.current_file_path}")

    def _close_and_upload_segment(self):
        """현재 세그먼트 완료 및 S3 업로드 + 백엔드 웹훅 등록"""
        if self.current_writer is not None:
            self.current_writer.release()
            self.current_writer = None

        if self.current_file_path and os.path.exists(self.current_file_path):
            file_to_upload = self.current_file_path
            start_time_iso = self.current_start_time.isoformat().replace("+00:00", "Z")
            end_time_dt = self.current_start_time + timedelta(seconds=(self.frame_count / self.fps))
            end_time_iso = end_time_dt.isoformat().replace("+00:00", "Z")
            filename = os.path.basename(file_to_upload)

            # 1. S3 업로드 (raw-videos/ 폴더)
            s3_key = f"raw-videos/{filename}"
            s3_url = upload_file_to_s3(file_to_upload, s3_key)
            print(f"[RawVideoSplitter] Zone {self.zone_id} 1분 영상 S3 업로드 완료: {s3_url}")

            # 2. Java 백엔드 /api/v1/video-clips 등록
            headers = {"Content-Type": "application/json", "X-API-KEY": BACKEND_API_KEY}
            clip_payload = {
                "zoneId": self.zone_id,
                "clipType": "TEMP",
                "s3ClipUrl": s3_url,
                "startTime": start_time_iso,
                "endTime": end_time_iso
            }
            try:
                backend_clip_url = f"{BACKEND_URL}/api/v1/video-clips"
                res = requests.post(backend_clip_url, headers=headers, json=clip_payload, timeout=5)
                print(f"[RawVideoSplitter] BE DB 적재 응답: {res.status_code}")
            except Exception as be_err:
                print(f"[RawVideoSplitter Warning] BE DB 적재 실패 ({be_err})")

            # 로컬 임시 파일 정리 (선택적)
            try:
                if os.path.exists(file_to_upload):
                    os.remove(file_to_upload)
            except Exception:
                pass

        self.current_file_path = None
        self.frame_count = 0

    def write_frame(self, frame):
        """매 프레임 기록 및 1분 도달 시 분할/업로드"""
        with self._lock:
            if self.current_writer is None:
                self._open_new_segment(frame)

            if self.target_size and frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(frame, self.target_size)

            self.current_writer.write(frame)
            self.frame_count += 1

            # 1분(60초) 분량 채웠을 때 스플릿
            if self.frame_count >= self.max_frames_per_split:
                self._close_and_upload_segment()

    def finalize(self):
        """스트림 종료 시 남아있는 버퍼 파일 업로드"""
        with self._lock:
            if self.frame_count > 0:
                self._close_and_upload_segment()
