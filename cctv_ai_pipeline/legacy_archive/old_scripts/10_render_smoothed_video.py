# 10_render_smoothed_video.py
# ===================================================================================================
# [목적 및 역할]
# 본 스크립트는 튐 현상이 보정(Smoothing)된 보행자 좌표 JSON 데이터(pedestrian_pixels_by_frame.json 및
# pedestrian_bev_xyz_by_frame.json)를 읽어서, CCTV 원본 프레임 이미지 위에
# 1) CCTV 2D 픽셀 좌표 오버레이 (보행자 ID & 위치 바운딩 포인트)
# 2) BEV 3D 미니맵 (Top-down view) 부드러운 위치 궤적 시각화
# 를 수행하고 고화질 웹 재생 호환 비디오 (cctv_smoothed_simulation_video.mp4)를 생성합니다.
# ===================================================================================================

import os
import sys
import glob
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
FRAMES_DIR = os.path.join(RESULTS_DIR, "frames")

# 1. 보정된 JSON 데이터 파일 경로
PIXELS_JSON_PATH = os.path.join(RESULTS_DIR, "pedestrian_pixels_by_frame.json")
BEV_JSON_PATH = os.path.join(RESULTS_DIR, "pedestrian_bev_xyz_by_frame.json")
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "cctv_smoothed_simulation_video.mp4")

if not os.path.exists(PIXELS_JSON_PATH) or not os.path.exists(BEV_JSON_PATH):
    print(f"[ERROR] 보정된 JSON 데이터 파일을 찾을 수 없습니다.")
    sys.exit(1)

with open(PIXELS_JSON_PATH, "r", encoding="utf-8") as f:
    pixels_by_frame = json.load(f)

with open(BEV_JSON_PATH, "r", encoding="utf-8") as f:
    bev_by_frame = json.load(f)

# 프레임 파일 탐색
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
if not frame_files:
    print(f"[ERROR] 프레임 이미지를 찾을 수 없습니다: {FRAMES_DIR}")
    sys.exit(1)

first_img = cv2.imread(frame_files[0])
img_h, img_w, _ = first_img.shape
print(f"[INFO] 원본 CCTV 해상도: {img_w}x{img_h}")

# 비디오 라이터 설정 (avc1 / H.264 재생 호환)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
try:
    fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc_h264, 10.0, (img_w, img_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 10.0, (img_w, img_h))
except Exception:
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 10.0, (img_w, img_h))

# BEV 미니맵 설정
minimap_w, minimap_h = 360, 480
x_range, y_range = (-10.0, 15.0), (-5.0, 25.0)

def bev_to_minimap_px(x_m, y_m):
    mx = int((x_m - x_range[0]) / (x_range[1] - x_range[0]) * minimap_w)
    my = int((y_m - y_range[0]) / (y_range[1] - y_range[0]) * minimap_h)
    return mx, my

print(f"[INFO] 튐 보정 완료된 동영상 렌더링 시작 ({len(pixels_by_frame)}개 프레임)...")

# ID별 무작위 색상 사전
np.random.seed(42)
color_palette = {}
for i in range(1, 200):
    color_palette[str(i)] = (
        int(np.random.randint(50, 255)),
        int(np.random.randint(50, 255)),
        int(np.random.randint(50, 255))
    )

frame_keys = sorted([int(k) for k in pixels_by_frame.keys()])

for f_idx, frame_id in enumerate(frame_keys):
    img_path = frame_files[(frame_id - 1) % len(frame_files)]
    canvas = cv2.imread(img_path)
    if canvas is None:
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    str_frame_id = str(frame_id)
    pixels_map = pixels_by_frame.get(str_frame_id, {})
    bev_map = bev_by_frame.get(str_frame_id, {})

    # 1) BEV 미니맵 캔버스 생성 (우측 상단 렌더링)
    minimap = np.full((minimap_h, minimap_w, 3), (30, 25, 20), dtype=np.uint8)
    cv2.putText(minimap, "[BEV 3D Smoothed Plot]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # 미니맵 격자선
    for xm in range(int(x_range[0]), int(x_range[1]) + 1, 5):
        mx, _ = bev_to_minimap_px(xm, 0)
        cv2.line(minimap, (mx, 0), (mx, minimap_h), (50, 45, 40), 1)
    for ym in range(int(y_range[0]), int(y_range[1]) + 1, 5):
        _, my = bev_to_minimap_px(0, ym)
        cv2.line(minimap, (0, my), (minimap_w, my), (50, 45, 40), 1)

    # 2) 보행자 좌표 시각화 (CCTV 화면 및 미니맵)
    for pid, p_coord in pixels_map.items():
        px, py = int(p_coord[0]), int(p_coord[1])
        b_coord = bev_map.get(pid, [0.0, 0.0, 0.0])
        bx, by = float(b_coord[0]), float(b_coord[1])

        color = color_palette.get(pid, (0, 255, 0))

        # CCTV 화면 점 및 ID 오버레이
        cv2.circle(canvas, (px, py), 6, color, -1)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), 2)
        cv2.putText(canvas, f"ID:{pid}", (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(canvas, f"ID:{pid}", (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # BEV 미니맵 좌표 오버레이
        mx, my = bev_to_minimap_px(bx, by)
        if 0 <= mx < minimap_w and 0 <= my < minimap_h:
            cv2.circle(minimap, (mx, my), 5, color, -1)
            cv2.putText(minimap, pid, (mx + 6, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 미니맵을 CCTV 메인 캔버스 우측 상단에 오버레이
    canvas[40:40+minimap_h, img_w-minimap_w-40:img_w-40] = minimap
    cv2.rectangle(canvas, (img_w-minimap_w-40, 40), (img_w-40, 40+minimap_h), (255, 255, 255), 2)

    # 상단 메인 캡션
    cv2.putText(canvas, f"MANGWON MARKET SMART CCTV (Smoothed Pedestrian Trajectory) - Frame {frame_id}", 
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Active Pedestrians: {len(pixels_map)} persons", 
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    writer.write(canvas)

writer.release()
print(f"\n============================================================")
print(f"[SUCCESS] 튐 보정 시각화 동영상 생성 완료!")
print(f" 저장 경로: {OUTPUT_VIDEO_PATH}")
print(f"============================================================")
