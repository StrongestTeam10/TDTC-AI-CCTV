# check_roi_exact_gt.py: 489개 JSON 정답 라벨 파일에 호모그래피 변환을 전수 적용하여 관심 영역(ROI) 거리 대역(10m, 15m, 20m)별 
# 정답 인원의 통계적 분포를 전수 조사하는 스크립트입니다.
import os
import json
import glob
import numpy as np

# 1. 경로 설정 및 호모그래피 행렬 (2D 픽셀 -> 3D BEV 미터 변환)
LABEL_DIR = r"E:\test\cctv_EXCO_test_label"
json_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))

H_MATRIX = np.array([
    [ 0.015, -0.002, -8.50],
    [ 0.001,  0.022, -2.10],
    [ 0.000,  0.001,  1.00]
])

def pixel_to_meter(u, v, H):
    pt = np.array([u, v, 1.0]).reshape(3, 1)
    bev_pt = np.dot(H, pt)
    bev_pt /= bev_pt[2]
    return float(bev_pt[0][0]), float(bev_pt[1][0]) # (X_m, Y_m)

total_gt = []
gt_10m = []
gt_15m = []
gt_20m = []

print(f"🔍 [489개 JSON 정답 라벨 전수 조사 시작]...\n")

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
    img_data = data.get('image', {})
    cnt_total = 0
    cnt_10 = 0
    cnt_15 = 0
    cnt_20 = 0
    
    if isinstance(img_data, dict):
        crowd = img_data.get('crowdinfo', {})
        if isinstance(crowd, dict):
            objs = crowd.get('objects', [])
            cnt_total = len(objs)
            
            for obj in objs:
                pt = obj.get('directionindex')
                if pt and len(pt) >= 2:
                    u, v = pt[0], pt[1]
                    x_m, y_m = pixel_to_meter(u, v, H_MATRIX)
                    
                    # 센서 앞쪽 거리별 분류 (가로 ±10m 내)
                    if -10.0 <= x_m <= 10.0:
                        if 0.0 <= y_m <= 10.0:
                            cnt_10 += 1
                        if 0.0 <= y_m <= 15.0:
                            cnt_15 += 1
                        if 0.0 <= y_m <= 20.0:
                            cnt_20 += 1
                            
    total_gt.append(cnt_total)
    gt_10m.append(cnt_10)
    gt_15m.append(cnt_15)
    gt_20m.append(cnt_20)

print("============================================================")
print("📊 [JSON 정답 라벨 실제 거리(ROI)별 전수 조사 결과]")
print("============================================================")
print(f"📄 총 검석 프레임: {len(json_files)} 개")
print(f"🌐 전체 정답 평균 (배경 구석 포함): {np.mean(total_gt):.1f} 명")
print("------------------------------------------------------------")
print(f"📍 10m 이내 유효 관제 구역 정답 평균: {np.mean(gt_10m):.1f} 명")
print(f"📍 15m 이내 유효 관제 구역 정답 평균: {np.mean(gt_15m):.1f} 명")
print(f"📍 20m 이내 유효 관제 구역 정답 평균: {np.mean(gt_20m):.1f} 명")
print("============================================================")