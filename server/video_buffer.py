"""
server/video_buffer.py - 위험 발생 시 전후 35초 클립 생성을 위한 원형 프레임 버퍼 (Rolling Buffer)
"""

import os
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional


class ZoneFrameBuffer:
    """단일 구역(Zone)에 대한 원형 프레임 버퍼"""

    def __init__(self, zone_id: int, max_seconds: float = 40.0, fps: float = 10.0):
        self.zone_id = zone_id
        self.max_seconds = max_seconds
        self.fps = fps
        self.max_frames = int(max_seconds * fps)
        # deque 내부에 (timestamp, frame_image) 튜플 보관
        self.buffer = deque(maxlen=self.max_frames)

    def append_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """새 프레임을 버퍼에 추가"""
        ts = timestamp or time.time()
        # 프레임 복사본 저장
        self.buffer.append((ts, frame.copy()))

    def get_buffered_duration(self) -> float:
        """현재 버퍼에 저장된 영상 길이(초)"""
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1][0] - self.buffer[0][0]

    def extract_clip(self, output_path: str, duration_sec: float = 35.0) -> Optional[str]:
        """
        현재 시점 기준 최근 `duration_sec` 분량의 프레임을 추출하여 MP4 비디오 파일로 저장합니다.
        """
        if not self.buffer:
            print(f"[VideoBuffer Warning] Zone {self.zone_id} 버퍼에 프레임이 없습니다.")
            return None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        frames_to_write = list(self.buffer)
        needed_frames = int(duration_sec * self.fps)
        if len(frames_to_write) > needed_frames:
            frames_to_write = frames_to_write[-needed_frames:]

        first_frame = frames_to_write[0][1]
        h, w = first_frame.shape[:2]

        # 코덱 탐색 및 VideoWriter 생성
        out_video = None
        for codec in ['mp4v', 'avc1', 'XVID']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_video = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
                if out_video.isOpened():
                    break
            except Exception:
                pass

        if out_video is None or not out_video.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        if not out_video.isOpened():
            print(f"[VideoBuffer Error] 비디오 출력 파일 생성 실패: {output_path}")
            return None

        try:
            for _, frm in frames_to_write:
                # 해상도 일치 확인
                if frm.shape[:2] != (h, w):
                    frm = cv2.resize(frm, (w, h))
                out_video.write(frm)
            out_video.release()
            print(f"[VideoBuffer] Zone {self.zone_id} 35초 위험 클립 추출 완료 ({len(frames_to_write)} 프레임) -> {output_path}")
            return output_path
        except Exception as e:
            print(f"[VideoBuffer Error] 프레임 인코딩 중 에러: {e}")
            if out_video is not None:
                out_video.release()
            return None

    def get_latest_snapshot(self) -> Optional[np.ndarray]:
        """버퍼에서 가장 최근 프레임 1장 반환"""
        if self.buffer:
            return self.buffer[-1][1].copy()
        return None


class MultiZoneBufferManager:
    """1, 2, 3구역 다중 프레임 버퍼 관리자"""

    def __init__(self, fps: float = 10.0, max_seconds: float = 40.0):
        self.fps = fps
        self.max_seconds = max_seconds
        self.buffers: Dict[int, ZoneFrameBuffer] = {
            1: ZoneFrameBuffer(zone_id=1, max_seconds=max_seconds, fps=fps),
            2: ZoneFrameBuffer(zone_id=2, max_seconds=max_seconds, fps=fps),
            3: ZoneFrameBuffer(zone_id=3, max_seconds=max_seconds, fps=fps),
        }

    def get_buffer(self, zone_id: int) -> ZoneFrameBuffer:
        if zone_id not in self.buffers:
            self.buffers[zone_id] = ZoneFrameBuffer(zone_id, self.max_seconds, self.fps)
        return self.buffers[zone_id]

    def push_frame(self, zone_id: int, frame: np.ndarray, timestamp: Optional[float] = None):
        buf = self.get_buffer(zone_id)
        buf.append_frame(frame, timestamp)

    def extract_danger_clip(self, zone_id: int, output_path: str, duration_sec: float = 35.0) -> Optional[str]:
        buf = self.get_buffer(zone_id)
        return buf.extract_clip(output_path, duration_sec)

    def save_snapshot(self, zone_id: int, output_image_path: str) -> Optional[str]:
        buf = self.get_buffer(zone_id)
        snap = buf.get_latest_snapshot()
        if snap is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
            cv2.imwrite(output_image_path, snap)
            return output_image_path
        return None


# 전역 다중 구역 버퍼 매니저 싱글톤 인스턴스
global_buffer_manager = MultiZoneBufferManager()
