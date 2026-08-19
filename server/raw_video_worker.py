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
from utils.db_connector import (
    get_db_connection,
    bulk_insert_video_clip,
    bulk_insert_pedestrian_coordinate_json
)


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
        self.buffered_metrics = []
        print(f"[RawVideoSplitter] Zone {self.zone_id} 새 1분 세그먼트 생성 시작: {self.current_file_path}")

    def _close_and_upload_segment(self):
        """현재 세그먼트 완료 및 S3 업로드 + RDS vdoclip01m & pedaggr01h & mrkrisk01m 직접 적재"""
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

                # 2. RDS vdoclip01m 테이블에 직접 INSERT 및 clip_id 획득
                created_clip_id = 1
                try:
                    conn = get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT INTO vdoclip01m (zone_id, factor_id, clip_type, s3_clip_url, start_time, end_time, is_downloaded, is_deleted, expires_at)
                            VALUES (%s, 1, 'TEMP', %s, %s, %s, FALSE, FALSE, NOW() + INTERVAL '1 hour')
                            RETURNING clip_id;
                        """, (self.zone_id, s3_url, start_time_iso, end_time_iso))
                        res_row = cur.fetchone()
                        if res_row:
                            created_clip_id = res_row[0]
                        conn.commit()
                        cur.close()
                        conn.close()
                        print(f"[RawVideoSplitter] RDS vdoclip01m 직접 적재 완료: clip_id = {created_clip_id}")
                except Exception as db_clip_err:
                    print(f"[RawVideoSplitter Warning] vdoclip01m 직접 적재 실패 ({db_clip_err})")

                # 3. 지난 1분간의 프레임 집계 데이터를 pedaggr01h & mrkrisk01m 테이블에 자동 적재
                if metrics_snapshot:
                    try:
                        for m in metrics_snapshot:
                            m["clip_id"] = created_clip_id
                            m["s3_clip_url"] = s3_url
                        bulk_insert_pedestrian_coordinate_json(metrics_snapshot)
                        print(f"[RawVideoSplitter] Zone {self.zone_id} 1분 집계 데이터 DB 적재 완료: {len(metrics_snapshot)} frames")
                    except Exception as db_err:
                        print(f"[RawVideoSplitter Warning] pedaggr01h / mrkrisk01m DB 적재 중 에러: {db_err}")

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

            # 10초(100프레임)마다 녹화 진행 상황 로그 출력
            if self.frame_count % 100 == 0:
                print(f"[RawVideoSplitter] Zone {self.zone_id} 1분 녹화 진행 중: {self.frame_count}/{self.max_frames_per_split} frames ({self.frame_count/self.fps:.0f}s / {self.split_duration_sec:.0f}s)")

            # 1분(60초 = 600프레임) 도달 시 분할 및 S3 + DB 적재
            if self.frame_count >= self.max_frames_per_split:
                print(f"[RawVideoSplitter] Zone {self.zone_id} 60초(600프레임) 도달 -> S3 및 DB 적재 트리거!")
                self._close_and_upload_segment()

    def finalize(self):
        """스트림 종료 시 남아있는 버퍼 파일 업로드"""
        with self._lock:
            if self.frame_count > 0:
                self._close_and_upload_segment()
