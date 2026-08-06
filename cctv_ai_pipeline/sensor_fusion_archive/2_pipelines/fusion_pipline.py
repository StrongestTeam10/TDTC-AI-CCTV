# fusion_pipline.py: YOLO(CCTV)와 PointPillar(LiDAR) 객체 검출 모델을 사용해 인원 카운팅 및 비교 교차 검증을 수행하는 초기 센서 퓨전 파이프라인입니다.
import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# =========================================================================
# 1. 경로 설정 (E 드라이브 기반)
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = r"E:\test"

YOLO_WEIGHT_PATH = os.path.join(RESULTS_DIR, "bestYOLOm5080model.pt")
POINTPILLAR_WEIGHT_PATH = os.path.join(RESULTS_DIR, "10_pointpillar_counter.pth")

CCTV_IMG_DIR = os.path.join(DATA_DIR, "cctv_EXCO_test")
CCTV_LABEL_DIR = os.path.join(DATA_DIR, "cctv_EXCO_test_label")

# =========================================================================
# 2. PointPillar 모델 구조 정의 (저장된 .pth 불러오기용)
# =========================================================================
class PointPillarPeopleCounter(nn.Module):
    def __init__(self, x_range=(-15, 15), y_range=(-15, 25), grid_size=0.5):
        super(PointPillarPeopleCounter, self).__init__()
        self.x_range, self.y_range, self.grid_size = x_range, y_range, grid_size
        self.nx = int((x_range[1] - x_range[0]) / grid_size)
        self.ny = int((y_range[1] - y_range[0]) / grid_size)

        self.pillar_net = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x_points):
        batch_size = x_points.size(0)
        device = x_points.device
        canvas = torch.zeros((batch_size, 64, self.ny, self.nx), device=device)

        for b in range(batch_size):
            pts = x_points[b]
            x_idx = ((pts[:, 0] - self.x_range[0]) / self.grid_size).long()
            y_idx = ((pts[:, 1] - self.y_range[0]) / self.grid_size).long()

            mask = (x_idx >= 0) & (x_idx < self.nx) & (y_idx >= 0) & (y_idx < self.ny)
            if not mask.any(): continue

            feat = self.pillar_net(pts[mask])
            pillar_indices = y_idx[mask] * self.nx + x_idx[mask]
            pillar_canvas = torch.zeros((64, self.ny * self.nx), device=device)
            pillar_canvas.index_add_(1, pillar_indices, feat.t())
            canvas[b] = pillar_canvas.view(64, self.ny, self.nx)

        features = self.backbone(canvas).view(batch_size, -1)
        return F.softplus(self.fc(features))

# =========================================================================
# 3. 모델 로드 및 추론 테스트
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ 사용 디바이스: {device}")

# A. YOLO CCTV 모델 로드
print(f"📦 YOLO 모델 로딩 중: {YOLO_WEIGHT_PATH}")
yolo_model = YOLO(YOLO_WEIGHT_PATH)

# B. PointPillar 모델 로드
print(f"📦 PointPillar 모델 로딩 중: {POINTPILLAR_WEIGHT_PATH}")
pillar_model = PointPillarPeopleCounter().to(device)
if os.path.exists(POINTPILLAR_WEIGHT_PATH):
    pillar_model.load_state_dict(torch.load(POINTPILLAR_WEIGHT_PATH, map_location=device))
    pillar_model.eval()
    print("✅ PointPillar 가중치 성공적으로 로드 완료!")
else:
    print(f"⚠️ 경고: {POINTPILLAR_WEIGHT_PATH} 경로에 파일이 없습니다.")

# =========================================================================
# 4. CCTV 이미지 데이터 테스트 및 추론 실행
# =========================================================================
image_files = sorted(glob.glob(os.path.join(CCTV_IMG_DIR, "*.jpg")))
print(f"\n🔍 [CCTV 이미지] 총 {len(image_files)}개 발견됨.")

if image_files:
    test_img_path = image_files[0]
    print(f"📸 테스트 이미지: {os.path.basename(test_img_path)}")

    # YOLO 추론
    results = yolo_model(test_img_path)
    
    # 2D Bounding Box 및 발밑 좌표 추출
    feet_coords = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            foot_x = (x1 + x2) / 2
            foot_y = y2  # 사람이 땅에 서 있는 발밑 점
            feet_coords.append((foot_x, foot_y))

    print(f"✅ CCTV 추론 완료: 총 {len(feet_coords)}명 감지됨")
    print(f"📍 추출된 발밑 2D 좌표 (샘플 3개): {feet_coords[:3]}")



# =========================================================================
# 5. 호모그래피(Homography) 매핑 및 BEV 시각화 함수
# =========================================================================
# EXCO 테스트용 임시 호모그래피 행렬 (실제 카메라-라이다 보정값에 맞춰 조정 가능)
# [2D Image Pixel (u, v, 1)] -> [3D BEV Ground (X, Y, 1)]
H_MATRIX = np.array([
    [ 0.015, -0.002, -8.50],
    [ 0.001,  0.022, -2.10],
    [ 0.000,  0.001,  1.00]
])

def cctv_to_bev(foot_coords, H):
    """CCTV 2D 발밑 좌표를 BEV 2D 공간 좌표로 변환"""
    bev_coords = []
    for (u, v) in foot_coords:
        pt = np.array([u, v, 1.0]).reshape(3, 1)
        bev_pt = np.dot(H, pt)
        bev_pt /= bev_pt[2]  # Normalize
        bev_coords.append((bev_pt[0][0], bev_pt[1][0]))
    return np.array(bev_coords)

# CCTV 좌표를 BEV 좌표로 변환
cctv_bev_points = cctv_to_bev(feet_coords, H_MATRIX)

# =========================================================================
# 6. 센서 퓨전 결과 시각화 (BEV Map 저장)
# =========================================================================
plt.figure(figsize=(10, 10))
plt.title(f"Sensor Fusion BEV Map - Detected People: {len(feet_coords)}", fontsize=14)

# CCTV 추정 위치 (파란색)
if len(cctv_bev_points) > 0:
    plt.scatter(cctv_bev_points[:, 0], cctv_bev_points[:, 1], 
                c='blue', label='CCTV Detection (YOLO)', s=60, alpha=0.7, edgecolors='k')

# 차후 LiDAR .bin 연결 시 포인트 추가 위치 (빨간색 예시)
# plt.scatter(lidar_x, lidar_y, c='red', label='LiDAR Cluster (DBSCAN)', s=80, marker='X')

plt.xlim(-15, 15)
plt.ylim(0, 30)
plt.xlabel("X (Meters) - Left/Right")
plt.ylabel("Y (Meters) - Forward Depth")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# 결과 이미지 저장
output_plot_path = os.path.join(RESULTS_DIR, "bev_fusion_result.png")
plt.savefig(output_plot_path, dpi=300)
plt.close()

print(f"📊 BEV 센서 퓨전 지도 저장 완료: {output_plot_path}")