# test_accuracy.py: JSON 정답 라벨의 픽셀 좌표(directionindex)를 호모그래피 변환한 값과 YOLO 검출 BEV 좌표 간의 오차 거리를 
# 계산하여 변환 정확도를 평가하는 스크립트입니다.
import os
import json
import glob
import numpy as np
import pandas as pd

# =========================================================================
# 1. 경로 및 호모그래피 변환 설정
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "cctv_bev_coordinates_second.csv")
LABEL_DIR = r"E:\test\cctv_EXCO_test_label"

# 호모그래피 행렬 (2D 픽셀 -> 3D BEV 미터 좌표)
H_MATRIX = np.array([
    [ 0.015, -0.002, -8.50],
    [ 0.001,  0.022, -2.10],
    [ 0.000,  0.001,  1.00]
])

def transform_pixel_to_bev(u, v, H):
    """2D 픽셀 좌표 (u, v) -> 라이다 BEV (X, Y) 미터 좌표 변환"""
    pt = np.array([u, v, 1.0]).reshape(3, 1)
    bev_pt = np.dot(H, pt)
    bev_pt /= bev_pt[2]
    return float(bev_pt[0][0]), float(bev_pt[1][0])

if not os.path.exists(CSV_PATH):
    print("⚠️ cctv_bev_coordinates_second.csv 파일이 없습니다. video_to_bev.py를 먼저 실행해 주세요.")
    exit()

cctv_df = pd.read_csv(CSV_PATH)
json_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))

print(f"🔍 [정밀 검증 시작] 총 {len(json_files)}개 JSON 정답과 호모그래피 오차 분석 중...\n")

def extract_exco_gt_bev_coords(label_data, H):
    """EXCO 데이터셋 전용: image -> crowdinfo -> objects -> directionindex 파싱"""
    gt_bev_coords = []
    
    img_data = label_data.get('image', {})
    if isinstance(img_data, dict):
        crowd_info = img_data.get('crowdinfo', {})
        if isinstance(crowd_info, dict):
            objs = crowd_info.get('objects', [])
            for obj in objs:
                # 💡 [핵심] directionindex 키 추출!
                pt = obj.get('directionindex')
                if pt and len(pt) >= 2:
                    # 정답 픽셀 좌표를 BEV 미터 좌표로 변환
                    bx, by = transform_pixel_to_bev(pt[0], pt[1], H)
                    gt_bev_coords.append([bx, by])
                    
    return gt_bev_coords

total_errors = []
gt_found_count = 0

for idx, json_path in enumerate(json_files, start=1):
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        label_data = json.load(f)
    
    gt_bev_pts = extract_exco_gt_bev_coords(label_data, H_MATRIX)
    pred_bev_pts = cctv_df[cctv_df['frame'] == idx][['bev_x_m', 'bev_y_m']].to_numpy()
    
    if len(gt_bev_pts) > 0:
        gt_found_count += 1
        
    if len(gt_bev_pts) > 0 and len(pred_bev_pts) > 0:
        gt_arr = np.array(gt_bev_pts)
        frame_errors = []
        for p_pt in pred_bev_pts:
            dists = np.linalg.norm(gt_arr - p_pt, axis=1)
            frame_errors.append(np.min(dists))
        total_errors.append(np.mean(frame_errors))

# =========================================================================
# 2. 최종 결과 출력
# =========================================================================
print("============================================================")
print("🎯 [호모그래피 좌표 변환 EXCO 데이터셋 최종 검증 결과]")
print("============================================================")
print(f"📄 정답 좌표 추출 성공 프레임: {gt_found_count} / {len(json_files)}")

if total_errors:
    mean_mae = np.mean(total_errors)
    print(f"📏 전체 평균 공간 거리 오차 (MAE): ±{mean_mae:.2f} m (미터)")
    print("============================================================")
    if mean_mae <= 1.5:
        print("🎉 [성공] 평균 오차가 매우 정밀하여 CCTV - LiDAR 센서 퓨전에 완벽 적용 가능합니다!")
    else:
        print(f"📊 평균 위치 오차: ±{mean_mae:.2f}m 도출 완료!")