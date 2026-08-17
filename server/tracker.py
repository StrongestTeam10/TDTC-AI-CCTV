"""
server/tracker.py - 비디오 프레임 흔들림 보정기 및 보행자 CentroidTracker 모듈
"""

import os
import tempfile
import shutil
import cv2
import numpy as np
from typing import Dict, Optional


def get_safe_video_capture(video_path: str):
    """한글/유니코드 경로를 안전하게 처리하는 VideoCapture 헬퍼"""
    is_unicode = False
    try:
        video_path.encode('ascii')
    except UnicodeEncodeError:
        is_unicode = True

    if is_unicode and os.path.exists(video_path):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_read_{os.getpid()}_{np.random.randint(1000)}.mp4")
        shutil.copy(video_path, temp_path)
        return cv2.VideoCapture(temp_path), temp_path
    else:
        return cv2.VideoCapture(video_path), None


class FrameStabilizer:
    """비디오 프레임 오프셋 역투영 흔들림 보정기"""
    def __init__(self, ground_roi=None):
        self.ref_frame = None
        self.ref_kp = None
        self.ref_des = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_H = np.eye(3, dtype=np.float32)
        self.fallback_count = 0
        self.max_fallback = 10
        self.ground_roi = ground_roi

    def initialize(self, frame):
        self.ref_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask = None
        if self.ground_roi is not None and len(self.ground_roi) >= 3:
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(mask, [self.ground_roi], 255)
            
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, mask)
        self.last_H = np.eye(3, dtype=np.float32)
        self.fallback_count = 0

    def stabilize_point(self, u, v, current_frame):
        if self.ref_des is None or self.ref_kp is None:
            self.initialize(current_frame)
            return (u, v), False

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        mask = None
        if self.ground_roi is not None and len(self.ground_roi) >= 3:
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(mask, [self.ground_roi], 255)

        kp, des = self.orb.detectAndCompute(gray, mask)

        if des is None or len(des) < 10:
            self.fallback_count += 1
            low_conf = self.fallback_count > self.max_fallback
            pt = np.array([u, v, 1.0]).reshape(3, 1)
            stab_pt = np.dot(self.last_H, pt)
            stab_pt /= stab_pt[2]
            return (float(stab_pt[0][0]), float(stab_pt[1][0])), low_conf

        matches = self.bf.match(self.ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        if len(good_matches) < 8:
            self.fallback_count += 1
            low_conf = self.fallback_count > self.max_fallback
            pt = np.array([u, v, 1.0]).reshape(3, 1)
            stab_pt = np.dot(self.last_H, pt)
            stab_pt /= stab_pt[2]
            return (float(stab_pt[0][0]), float(stab_pt[1][0])), low_conf

        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, inliers = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            self.fallback_count += 1
            low_conf = self.fallback_count > self.max_fallback
            pt = np.array([u, v, 1.0]).reshape(3, 1)
            stab_pt = np.dot(self.last_H, pt)
            stab_pt /= stab_pt[2]
            return (float(stab_pt[0][0]), float(stab_pt[1][0])), low_conf

        self.last_H = H
        self.fallback_count = 0

        pt = np.array([u, v, 1.0]).reshape(3, 1)
        stab_pt = np.dot(H, pt)
        stab_pt /= stab_pt[2]
        return (float(stab_pt[0][0]), float(stab_pt[1][0])), False


class CentroidTracker:
    """
    보행자 중심점 기반 ID 지속성 추적 및 정체 시간(Stagnation) 계산기
    """
    def __init__(self, max_disappeared: int = 10, max_distance: float = 60.0):
        self.next_object_id = 0
        self.objects: Dict[int, tuple] = {}
        self.disappeared: Dict[int, int] = {}
        self.tracks: Dict[int, list] = {}
        self.stagnant_frames: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = []
        self.stagnant_frames[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id: int):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.tracks.pop(object_id, None)
        self.stagnant_frames.pop(object_id, None)

    def update(self, rects: list, frame_id: int, min_move_thresh: float = 8.0):
        """
        검출 좌표 목록을 수신하여 트래킹 ID를 매핑하고 이동 거리를 기반으로 정체 프레임을 누적합니다.
        """
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array(rects)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                obj_id = self.register(input_centroids[i])
                self.tracks[obj_id].append((frame_id, input_centroids[i][0], input_centroids[i][1]))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                new_pt = input_centroids[col]
                prev_pt = self.objects[object_id]

                # 이동 거리(Displacement) 계산
                move_dist = np.linalg.norm(np.array(new_pt) - np.array(prev_pt))
                if move_dist < min_move_thresh:
                    self.stagnant_frames[object_id] = self.stagnant_frames.get(object_id, 0) + 1
                else:
                    self.stagnant_frames[object_id] = max(0, self.stagnant_frames.get(object_id, 0) - 2)

                self.objects[object_id] = new_pt
                self.disappeared[object_id] = 0
                self.tracks[object_id].append((frame_id, new_pt[0], new_pt[1]))
                if len(self.tracks[object_id]) > 10:
                    self.tracks[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])).difference(used_rows):
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in set(range(D.shape[1])).difference(used_cols):
                obj_id = self.register(input_centroids[col])
                self.tracks[obj_id].append((frame_id, input_centroids[col][0], input_centroids[col][1]))

        return self.objects

    def get_stagnation_metrics(self, fps: float = 10.0) -> Dict[str, float]:
        """현재 활성 트랙들의 평균 및 최대 정체 시간(초) 반환"""
        if not self.objects or not self.stagnant_frames:
            return {"avg_stagnation_sec": 0.0, "max_stagnation_sec": 0.0}

        active_stag_secs = [
            self.stagnant_frames.get(obj_id, 0) / max(1.0, fps)
            for obj_id in self.objects.keys()
        ]
        return {
            "avg_stagnation_sec": round(float(np.mean(active_stag_secs)), 1),
            "max_stagnation_sec": round(float(np.max(active_stag_secs)), 1)
        }
