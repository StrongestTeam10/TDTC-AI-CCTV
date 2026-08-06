# 05_render_pedestrian_coords_video.py
# 실제 CCTV 카메라 원본 프레임(1920x1080) 위에 좌우 반전(Horizontal Flip) 보정된 픽셀 좌표(pixel_x, pixel_y)와 BEV 미터 좌표(bev_x_m, bev_y_m)를 100% 정합하여 동영상 생성하는 스크립트

import os
import glob
import cv2
import numpy as np
import pandas as pd

# 1. 경로 및 웹 호환 파일명 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 루트 results 폴더로 단일 통일
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
FRAMES_DIR = os.path.join(RESULTS_DIR, "frames")

HISTORY_CSV_PATH = os.path.join(RESULTS_DIR, "mangwon_label_pedestrians.csv")
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "cctv_simulation_video.mp4")

# 2. 데이터 및 프레임 이미지 목록 로드
print(f"[LOAD] Exact 1080p Label CSV: {HISTORY_CSV_PATH}")
if not os.path.exists(HISTORY_CSV_PATH):
    print(f"[ERROR] {HISTORY_CSV_PATH} file not found.")
    exit(1)

df_history = pd.read_csv(HISTORY_CSV_PATH)
if 'sequence_id' in df_history.columns:
    df_history = df_history[df_history['sequence_id'] == 1]

frame_indices = sorted(df_history['frame_id'].unique())

# 프레임 이미지 경로 수집
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
if not frame_files:
    print(f"[ERROR] No jpg frame images found in: {FRAMES_DIR}")
    exit(1)

first_img = cv2.imread(frame_files[0])
if first_img is None:
    print(f"[ERROR] Cannot read frame image: {frame_files[0]}")
    exit(1)

img_h, img_w, _ = first_img.shape
print(f"[INFO] CCTV Frame Resolution: {img_w}x{img_h}")

# 3. 웹 브라우저 재생(H.264) 호환 avc1 코덱 동영상 생성기 설정
fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = 10.0
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (img_w, img_h))

if not video_writer.isOpened():
    print(f"[ERROR] Cannot open VideoWriter: {OUTPUT_VIDEO_PATH}")
    exit(1)

# 미니맵 (PiP) BEV 맵 크기 설정
minimap_w, minimap_h = 320, 420
x_range, y_range = (-15.0, 15.0), (-15.0, 25.0)

def bev_to_minimap_px(x_m, y_m):
    """BEV 미터 좌표 -> 미니맵 픽셀 좌표 변환"""
    mx = int((x_m - x_range[0]) / (x_range[1] - x_range[0]) * minimap_w)
    my = int((y_m - y_range[0]) / (y_range[1] - y_range[0]) * minimap_h)
    return mx, my

# 2D 이미지 화면 상의 흰색 사다리꼴 ROI 다각형 정점 (FHD 1920x1080 기준)
# (보정된 픽셀 좌표 u_val, v_val 기준 사다리꼴 통로 내 포함 여부 판단)
ROI_2D_POLY = np.array([
    [70, 950],     # 좌하단
    [320, 240],    # 좌상단
    [920, 240],    # 우상단
    [1080, 860]    # 우하단
], dtype=np.int32)

print(f"[PROCESSING] Rendering Horizontal-Flipped Corrected CCTV ROI Overlay ({len(frame_indices)} frames)...")

for idx, frame_idx in enumerate(frame_indices):
    img_path = frame_files[(frame_idx - 1) % len(frame_files)]
    frame_img = cv2.imread(img_path)
    if frame_img is None:
        continue

    canvas = frame_img.copy()
    frame_df = df_history[df_history['frame_id'] == frame_idx]

    # 1) BEV 미니맵 (Picture-in-Picture) 캔버스 생성
    minimap = np.full((minimap_h, minimap_w, 3), (25, 20, 15), dtype=np.uint8)
    cv2.putText(minimap, "BEV Mini-Map (m)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    
    # 미니맵 그리드
    for xm in range(int(x_range[0]), int(x_range[1]) + 1, 10):
        mx, _ = bev_to_minimap_px(xm, 0)
        cv2.line(minimap, (mx, 0), (mx, minimap_h), (40, 35, 30), 1)
    for ym in range(int(y_range[0]), int(y_range[1]) + 1, 10):
        _, my = bev_to_minimap_px(0, ym)
        cv2.line(minimap, (0, my), (minimap_w, my), (40, 35, 30), 1)

    # 2D 픽셀 ROI 통로 내부 보행자만 필터링
    roi_rows = []
    for _, row in frame_df.iterrows():
        raw_u = float(row['pixel_x'])
        u_val = img_w - raw_u
        v_val = float(row['pixel_y'])
        
        # 2D 다각형 내부에 위치하는지 검사 (포함 시 >= 0)
        in_2d_roi = cv2.pointPolygonTest(ROI_2D_POLY, (u_val, v_val), False) >= 0
        if in_2d_roi:
            roi_rows.append(row)
            
    filtered_df = pd.DataFrame(roi_rows) if roi_rows else pd.DataFrame()

    # 2) 상단 상태 헤더 바 렌더링
    cv2.rectangle(canvas, (0, 0), (img_w, 55), (0, 0, 0), -1)
    cv2.putText(canvas, f"REAL CCTV ROI PEDESTRIAN OVERLAY - Frame #{frame_idx}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Strict ROI Peoples: {len(filtered_df)}", (img_w - 320, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

    # 3) 2D ROI 내부 보행자 오버레이 및 라벨 렌더링
    for _, row in filtered_df.iterrows():
        p_id = int(row['person_id'])
        
        # 💡 좌우 반전 보정 (img_w - pixel_x)
        raw_u = float(row['pixel_x'])
        u_val = img_w - raw_u
        v_val = float(row['pixel_y'])
        
        bev_x = float(row['bev_x_m'])
        bev_y = float(row['bev_y_m'])
        
        u_int, v_int = int(u_val), int(v_val)

        # 미니맵 상의 좌표 (X축 대칭 반전 보정)
        mx, my = bev_to_minimap_px(-bev_x, bev_y)
        if 0 <= mx < minimap_w and 0 <= my < minimap_h:
            cv2.circle(minimap, (mx, my), 4, (255, 140, 0), -1) # 네온 오렌지
            cv2.putText(minimap, f"P{p_id}", (mx + 5, my + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # CCTV 화면 범위 안일 경우 정확한 (u_val, v_val) 위치에 타겟 마크 및 라벨 표시
        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            # 보행자 위치 앵커 및 조준 서클 (네온 시안 + 핫 핑크)
            cv2.circle(canvas, (u_int, v_int), 6, (255, 255, 0), -1)    # 네온 시안 중심 점
            cv2.circle(canvas, (u_int, v_int), 12, (255, 108, 0), 2)  # 블루-오렌지 외경
            cv2.circle(canvas, (u_int, v_int), 16, (255, 0, 255), 1)  # 네온 핑크 조준선

            # 상세 좌표 라벨 텍스트
            label_title = f"ID:{p_id}"
            label_px = f"pixel: ({u_val:.1f}, {v_val:.1f})"
            label_bev = f"bev: ({bev_x:.2f}m, {bev_y:.2f}m)"

            font_scale = 0.42
            line_height = 16
            box_width = 190
            box_height = 54

            lx = min(max(u_int + 20, 10), img_w - box_width - 10)
            ly = min(max(v_int - 20, 65), img_h - box_height - 10)

            # 연결 지도 선 (Target Line)
            cv2.line(canvas, (u_int, v_int), (lx - 5, ly + 10), (255, 255, 0), 1)

            # 라벨 배경 상자 및 테두리 (네온 스타일)
            cv2.rectangle(canvas, (lx - 5, ly - 15), (lx + box_width, ly + box_height - 15), (15, 15, 20), -1)
            cv2.rectangle(canvas, (lx - 5, ly - 15), (lx + box_width, ly + box_height - 15), (255, 255, 0), 1)

            # 텍스트 라인 렌더링
            cv2.putText(canvas, label_title, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, label_px, (lx, ly + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, label_bev, (lx, ly + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 200), 1, cv2.LINE_AA)

    # 4) BEV 미니맵을 CCTV 우상단에 삽입 (Picture-in-Picture)
    margin = 15
    top_y = 65
    left_x = img_w - minimap_w - margin
    canvas[top_y:top_y + minimap_h, left_x:left_x + minimap_w] = minimap
    cv2.rectangle(canvas, (left_x, top_y), (left_x + minimap_w, top_y + minimap_h), (255, 255, 0), 2)

    video_writer.write(canvas)

video_writer.release()
print(f"[SUCCESS] Flipped-Corrected Video created successfully!")
print(f"[SAVED] Path: {OUTPUT_VIDEO_PATH}")
