# fusion_validation_v3.py
# ===================================================================================================
# [목적 및 역할]
# 본 스크립트는 CCTV(카메라) BEV 역투영 밀도 데이터와 PointPillar 기반의 LiDAR 딥러닝 모델 추론 결과를
# 2D BEV 격자(Grid) 평면 상에서 픽셀 단위로 직접 융합(Grid-level/Pixel-level Fusion)하는 검증 스크립트입니다.
# 
# [주요 기능]
# 1. 딥러닝 기반 LiDAR BEV 점유 모델(PointPillarCenterPointModel) 인스턴스화 및 가중치(lidar_AI.pth) 로드
# 2. CCTV BEV 좌표 CSV를 바탕으로 2D BEV 가우시안 점유 히트맵 생성 (160 x 120 격자)
# 3. 프레임별 시뮬레이션 포인트 클라우드 생성 후, LiDAR 모델 추론을 통해 격자 점유 히트맵 예측
# 4. 카메라 BEV 히트맵과 라이다 BEV 히트맵 간 격자 레벨 가중 융합(Weighted Fusion) 적용
# 5. 융합된 최종 히트맵 상에서 Peak 검출(Local Maxima Filter)을 적용하여 정확한 최종 인원수 산출
# ===================================================================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.ndimage as ndimage
# pyrefly: ignore [missing-import]
import cv2
from shapely.geometry import Polygon, Point

# utils 모듈 참조를 위해 sys.path에 부모 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyrefly: ignore [missing-import]
from utils.db_connector import bulk_insert_summary, bulk_insert_history, bulk_insert_roi_log
# pyrefly: ignore [missing-import]
from utils.s3_uploader import archive_local_csv_to_zip

# =========================================================================
# 1. 경로 및 하이퍼파라미터 설정
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CCTV_CSV_PATH = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv"))
OUTPUT_SUMMARY_PATH = os.environ.get("FUSION_SUMMARY_CSV", os.path.join(RESULTS_DIR, "sensor_fusion_summary_v2.csv"))
MODEL_WEIGHT_PATH = os.path.join(RESULTS_DIR, "lidar_AI.pth") # 또는 10_pointpillar_counter.pth

if not os.path.exists(CCTV_CSV_PATH):
    print(f"⚠️ CCTV BEV 좌표 CSV 파일이 존재하지 않습니다: {CCTV_CSV_PATH}")
    print("먼저 3단계: video_to_bev_CSR.py 를 실행하여 CSV를 생성하십시오.")
    sys.exit(1)

cctv_df = pd.read_csv(CCTV_CSV_PATH)
total_frames = int(cctv_df['frame'].max()) if 'frame' in cctv_df.columns else 489

print(f"🚀 [LiDAR 히트맵 + CCTV BEV 격자 레벨 퓨전 v3 시작] 총 {total_frames} 프레임 처리 중...\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 사용 디바이스: {device}")

# =========================================================================
# 2. LiDAR 3D PointPillar + CenterPoint 모델 정의 (1_ai_models/lidar_ai.py 이식)
# =========================================================================
class PointPillarCenterPointModel(nn.Module):
    def __init__(self, x_range=(-15, 15), y_range=(-15, 25), grid_size=0.25):
        super(PointPillarCenterPointModel, self).__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.nx = int((x_range[1] - x_range[0]) / grid_size) # 120
        self.ny = int((y_range[1] - y_range[0]) / grid_size) # 160

        # Point Feature Net
        self.pfn = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 2D Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Centerpoint Head (히트맵 출력용)
        self.center_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_points):
        # x_points: [B, N, 3] (Point Cloud [x, y, z])
        B, N, _ = x_points.shape
        device = x_points.device

        pts_flat = x_points.reshape(B * N, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        valid_mask = (pts_flat[:, 0] != 0) | (pts_flat[:, 1] != 0) | (pts_flat[:, 2] != 0)

        x_idx = ((pts_flat[:, 0] - self.x_range[0]) / self.grid_size).long()
        y_idx = ((pts_flat[:, 1] - self.y_range[0]) / self.grid_size).long()
        in_range = (x_idx >= 0) & (x_idx < self.nx) & (y_idx >= 0) & (y_idx < self.ny)

        mask = valid_mask & in_range
        if mask.sum() == 0:
            return torch.zeros((B, 1, self.ny, self.nx), device=device, dtype=x_points.dtype)

        pts_valid = pts_flat[mask]
        x_idx_v = x_idx[mask]
        y_idx_v = y_idx[mask]
        batch_idx_v = batch_idx[mask]

        pillar_center_x = self.x_range[0] + (x_idx_v.float() + 0.5) * self.grid_size
        pillar_center_y = self.y_range[0] + (y_idx_v.float() + 0.5) * self.grid_size

        xc = pts_valid[:, 0] - pillar_center_x
        yc = pts_valid[:, 1] - pillar_center_y
        zc = pts_valid[:, 2] - pts_valid[:, 2].mean()

        xp = pts_valid[:, 0] - pillar_center_x
        yp = pts_valid[:, 1] - pillar_center_y

        feat_in = torch.stack([
            pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2],
            xc, yc, zc, xp, yp
        ], dim=1)

        feat = self.pfn(feat_in)

        flat_pillar_idx = batch_idx_v * (self.ny * self.nx) + y_idx_v * self.nx + x_idx_v

        canvas_flat = torch.zeros((B * self.ny * self.nx, 64), device=device, dtype=feat.dtype)
        canvas_flat = canvas_flat.scatter_reduce(
            0,
            flat_pillar_idx.unsqueeze(1).expand(-1, 64),
            feat,
            reduce="amax",
            include_self=True
        )

        canvas = canvas_flat.view(B, self.ny, self.nx, 64).permute(0, 3, 1, 2).contiguous()
        bev_feat = self.backbone(canvas)
        heatmap = self.center_head(bev_feat)
        return heatmap

# =========================================================================
# 3. 모델 로드 및 초기화
# =========================================================================
model = PointPillarCenterPointModel().to(device)
if os.path.exists(MODEL_WEIGHT_PATH):
    print(f"📂 LiDAR 모델 가중치 로드 중: {MODEL_WEIGHT_PATH}")
    ckpt = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt, strict=False)
    model.eval()
    print("✅ LiDAR 모델 가중치 로드 완료!")
else:
    print(f"⚠️ LiDAR 가중치 파일({MODEL_WEIGHT_PATH})을 찾을 수 없습니다. 랜덤 가중치로 추론을 시뮬레이션합니다.")
    model.eval()

# =========================================================================
# 4. 보조 함수 정의 (CCTV 가우시안 생성, 피크 검출 등)
# =========================================================================
def create_cctv_gaussian_heatmap(c_sub, nx=120, ny=160, x_range=(-15, 15), y_range=(-15, 25), grid_size=0.25, radius=3, sigma=1.2):
    """CCTV BEV 좌표 목록을 가우시안 히트맵으로 렌더링"""
    heatmap = np.zeros((ny, nx), dtype=np.float32)
    for _, row in c_sub.iterrows():
        cx, cy = row['bev_x_m'], row['bev_y_m']
        x_idx = int((cx - x_range[0]) / grid_size)
        y_idx = int((cy - y_range[0]) / grid_size)
        
        # 가우시안 렌더링
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                gx, gy = x_idx + dx, y_idx + dy
                if 0 <= gx < nx and 0 <= gy < ny:
                    val = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    heatmap[gy, gx] = max(heatmap[gy, gx], val)
    return heatmap

def extract_peaks_from_heatmap(heatmap, threshold=0.25):
    """융합 히트맵에서 로컬 피크를 찾아 객체 개수를 카운팅"""
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(heatmap, neighborhood_size)
    maxima = (heatmap == data_max)
    data_min = ndimage.minimum_filter(heatmap, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[~diff] = 0

    labeled, num_objects = ndimage.label(maxima)
    return num_objects

# =========================================================================
# 5. 위험 구역(ROI) 정의 및 사전 연산 마스크 생성
# =========================================================================
# ROI 구역 다각형 및 정원 제한 정의
roi_zones = [
    {
        "roi_id": 1,
        "roi_name": "Cafe_Entrance_Zone",
        "polygon": Polygon([(-2.0, 1.0), (2.0, 1.0), (2.0, 5.0), (-2.0, 5.0)]),
        "capacity_limit": 5
    },
    {
        "roi_id": 2,
        "roi_name": "Main_Counter_Zone",
        "polygon": Polygon([(-5.0, 5.0), (-1.0, 5.0), (-1.0, 10.0), (-5.0, 10.0)]),
        "capacity_limit": 8
    }
]

# 격자 파라미터 정의
nx, ny = 120, 160
x_range, y_range = (-15, 15), (-15, 25)
grid_size = 0.25
max_points = 8192

# ROI 마스크 사전 계산 (성능 최적화)
for zone in roi_zones:
    poly = zone["polygon"]
    mask = np.zeros((ny, nx), dtype=bool)
    for y_idx in range(ny):
        for x_idx in range(nx):
            cx = x_range[0] + x_idx * grid_size + 0.5 * grid_size
            cy = y_range[0] + y_idx * grid_size + 0.5 * grid_size
            if poly.contains(Point(cx, cy)):
                mask[y_idx, x_idx] = True
    zone["mask"] = mask

def extract_peak_coordinates(heatmap, threshold=0.3, x_range=(-15, 15), y_range=(-15, 25), grid_size=0.25):
    """융합 히트맵에서 로컬 피크를 찾아 물리 좌표(X, Y) 리스트를 추출"""
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(heatmap, neighborhood_size)
    maxima = (heatmap == data_max)
    data_min = ndimage.minimum_filter(heatmap, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[~diff] = 0
    
    y_indices, x_indices = np.where(maxima)
    coords = []
    for y_idx, x_idx in zip(y_indices, x_indices):
        cx = x_range[0] + x_idx * grid_size + 0.5 * grid_size
        cy = y_range[0] + y_idx * grid_size + 0.5 * grid_size
        coords.append((cx, cy))
    return coords

# =========================================================================
# 6. 프레임별 통합 퓨전 연산 및 위험도 스코어링 루프
# =========================================================================
summary_data = []
roi_risk_logs = []
pedestrian_history = []

# 퓨전 가중치 파라미터
ALPHA = 0.6  # CCTV 히트맵 가중치
BETA = 0.4   # LiDAR 히트맵 가중치

# 비디오 저장 설정 ( results/fusion_heatmap_movie.mp4 )
video_path = os.path.join(RESULTS_DIR, "fusion_heatmap_movie.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (nx * 3 * 3, ny * 3))

for frame_idx in range(1, total_frames + 1):
    # 1) 해당 프레임 CCTV 좌표 추출
    c_sub = cctv_df[cctv_df['frame'] == frame_idx]
    cctv_count = len(c_sub)
    
    # CCTV BEV 가우시안 히트맵 생성
    cctv_heatmap = create_cctv_gaussian_heatmap(c_sub, nx, ny, x_range, y_range, grid_size)

    # 2) LiDAR 가상 3D 점구름 데이터 시뮬레이션
    np.random.seed(frame_idx)
    simulated_points = []
    
    # CCTV가 감지한 영역 주위에 물리 점구름 배치
    for _, row in c_sub.iterrows():
        bx, by = row['bev_x_m'], row['bev_y_m']
        for _ in range(np.random.randint(5, 15)):
            px = bx + np.random.normal(0, 0.2)
            py = by + np.random.normal(0, 0.2)
            pz = np.random.uniform(0.1, 1.8)
            simulated_points.append([px, py, pz])
            
    # 사각지대에 위치한 보행자들을 위한 가상 점구름 배치 (추가 2~5명)
    extra_people = np.random.randint(2, 6)
    for _ in range(extra_people):
        ex = np.random.uniform(-10.0, 10.0)
        ey = np.random.uniform(2.0, 18.0)
        for _ in range(np.random.randint(6, 18)):
            simulated_points.append([ex + np.random.normal(0, 0.15), ey + np.random.normal(0, 0.15), np.random.uniform(0.1, 1.8)])
            
    points_arr = np.array(simulated_points) if len(simulated_points) > 0 else np.zeros((1, 3))
    
    # 점구름 개수 정규화 (8192개)
    num_pts = len(points_arr)
    if num_pts >= max_points:
        choice = np.random.choice(num_pts, max_points, replace=False)
        points_arr = points_arr[choice]
    else:
        if num_pts > 0:
            pad = np.zeros((max_points - num_pts, 3), dtype=np.float32)
            points_arr = np.vstack((points_arr, pad))
        else:
            points_arr = np.zeros((max_points, 3), dtype=np.float32)

    # 3) LiDAR 모델 추론 (BEV 히트맵 예측)
    pts_tensor = torch.tensor(points_arr, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        lidar_pred_heatmap = model(pts_tensor) # [1, 1, 160, 120]
        
    lidar_heatmap = lidar_pred_heatmap.squeeze().cpu().numpy()

    # 4) Grid-level 가중 센서 퓨전
    fusion_heatmap = ALPHA * cctv_heatmap + BETA * lidar_heatmap
    
    # 5) Peak 검출 기반 최종 인원 좌표 추출
    fused_coords = extract_peak_coordinates(fusion_heatmap, threshold=0.3, x_range=x_range, y_range=y_range, grid_size=grid_size)
    total_fusion_count = len(fused_coords)
    
    # CCTV가 관측한 인원은 강제로 보정하여 누락 차단
    if total_fusion_count < cctv_count:
        total_fusion_count = cctv_count
        fused_coords = [(row['bev_x_m'], row['bev_y_m']) for _, row in c_sub.iterrows()]
        
    # 라이다 모델 단독 카운트 계산 (비교용)
    lidar_only_count = extract_peaks_from_heatmap(lidar_heatmap, threshold=0.3)
    lidar_added = max(0, total_fusion_count - cctv_count)
    matched_count = min(cctv_count, max(0, total_fusion_count - cctv_count))

    # 6) 보행자 좌표 히스토리 적재용 데이터 저장
    for p_id, (cx, cy) in enumerate(fused_coords):
        pedestrian_history.append({
            'frame_id': frame_idx,
            'person_id': p_id,
            'bev_x_m': float(cx),
            'bev_y_m': float(cy),
            'detection_source': 'Fused'
        })

    # 7) 구역(ROI)별 위험도 스코어 계산
    max_risk_score = 0.0
    for zone in roi_zones:
        poly = zone["polygon"]
        people_in_roi = sum(1 for cx, cy in fused_coords if poly.contains(Point(cx, cy)))
        
        # 구역 격자의 평균 융합 밀도 강도
        occupancy_intensity = float(fusion_heatmap[zone["mask"]].mean()) if zone["mask"].sum() > 0 else 0.0
        
        # 위험 스코어 공식: (15 * 인원) + (50 * 평균 밀도 강도)
        risk_score = (15.0 * people_in_roi) + (50.0 * occupancy_intensity)
        risk_score = min(100.0, max(0.0, risk_score))
        max_risk_score = max(max_risk_score, risk_score)
        
        # 위험 등급 판정
        if risk_score < 30: risk_level = "SAFE"
        elif risk_score < 50: risk_level = "ATTENTION"
        elif risk_score < 75: risk_level = "CAUTION"
        elif risk_score < 90: risk_level = "WARNING"
        else: risk_level = "DANGER"
        
        roi_risk_logs.append({
            'frame_id': frame_idx,
            'roi_id': zone['roi_id'],
            'roi_name': zone['roi_name'],
            'people_in_roi': people_in_roi,
            'occupancy_intensity': occupancy_intensity,
            'risk_score': risk_score,
            'risk_level': risk_level
        })

    summary_data.append({
        'frame_id': frame_idx,
        'timestamp_sec': float(frame_idx / 10.0), # 10 FPS 가정
        'cctv_count': cctv_count,
        'lidar_added_count': lidar_added,
        'total_fusion_count': total_fusion_count,
        'max_risk_score': max_risk_score
    })

    # 실시간 프레임용 병합 이미지 생성 (비디오 입력용)
    cctv_scaled = (np.clip(cctv_heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    lidar_scaled = (np.clip(lidar_heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    fusion_scaled = (np.clip(fusion_heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    
    cctv_color = cv2.applyColorMap(cctv_scaled, cv2.COLORMAP_JET)
    lidar_color = cv2.applyColorMap(lidar_scaled, cv2.COLORMAP_JET)
    fusion_color = cv2.applyColorMap(fusion_scaled, cv2.COLORMAP_JET)
    
    for zone in roi_zones:
        poly = zone["polygon"]
        pts = np.array([[(x - x_range[0]) / grid_size, (y - y_range[0]) / grid_size] for x, y in poly.exterior.coords], dtype=np.int32)
        cv2.polylines(fusion_color, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
        
    combined = np.hstack((cctv_color, lidar_color, fusion_color))
    combined_resized = cv2.resize(combined, (nx * 3 * 3, ny * 3))
    
    cv2.putText(combined_resized, "CCTV Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_resized, "LiDAR Model Heatmap", (nx * 3 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_resized, f"Fused Heatmap (ROI Outline) - Frame {frame_idx}", (nx * 6 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 비디오 라이터에 프레임 기록
    video_writer.write(combined_resized)

    # 100프레임 간격마다 이미지 스냅샷도 백업 저장
    if frame_idx % 100 == 0:
        vis_dir = os.path.join(RESULTS_DIR, "heatmap_vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"fusion_heatmap_frame_{frame_idx}.png")
        cv2.imwrite(save_path, combined_resized)

# 비디오 라이터 닫기
video_writer.release()
print(f"[SUCCESS] 퓨전 히트맵 동영상 저장 완료: {video_path}")

# =========================================================================
# 8. 로컬 백업 CSV 저장 및 통계 출력
# =========================================================================

# 저장 폴더 구성
os.makedirs(RESULTS_DIR, exist_ok=True)

# csv 파일 적재
df_summary = pd.DataFrame(summary_data)
df_history = pd.DataFrame(pedestrian_history)
df_roi = pd.DataFrame(roi_risk_logs)

df_summary.to_csv(os.path.join(RESULTS_DIR, "sensor_fusion_summary_v3.csv"), index=False)
df_history.to_csv(os.path.join(RESULTS_DIR, "sf_pedestrian_history.csv"), index=False)
df_roi.to_csv(os.path.join(RESULTS_DIR, "sf_roi_risk_log.csv"), index=False)

print("============================================================")
print("🎯 [LiDAR Deep Learning Heatmap + ROI 위험도 분석 v3 완료]")
print("============================================================")
print(f"📁 프레임 요약 저장: {os.path.join(RESULTS_DIR, 'sensor_fusion_summary_v3.csv')}")
print(f"📁 보행자 좌표 저장: {os.path.join(RESULTS_DIR, 'sf_pedestrian_history.csv')}")
print(f"📁 ROI 위험 로그 저장: {os.path.join(RESULTS_DIR, 'sf_roi_risk_log.csv')}")
print(f"👥 프레임당 CCTV 평균 탐지: {df_summary['cctv_count'].mean():.1f} 명")
print(f"🚀 프레임당 최종 퓨전 통합 인원: {df_summary['total_fusion_count'].mean():.1f} 명")
print(f"🔥 최고 위험 영역 위험도 평균: {df_summary['max_risk_score'].mean():.2f} 점")
print("============================================================")

# =========================================================================
# 9. 클라우드 백엔드 적재 (S3 및 Supabase 연동)
# =========================================================================
print("\n[INFO] 클라우드 백엔드 데이터 적재 시작...")
# 1) Supabase DB 적재 (Bulk Insert)
summary_list = df_summary.to_dict(orient='records')
history_list = df_history.to_dict(orient='records')
roi_list = df_roi.to_dict(orient='records')

bulk_insert_summary(summary_list)
bulk_insert_history(history_list)
bulk_insert_roi_log(roi_list)

# 2) AWS S3 아카이빙 적재
csv_files = [
    os.path.join(RESULTS_DIR, "sensor_fusion_summary_v3.csv"),
    os.path.join(RESULTS_DIR, "sf_pedestrian_history.csv"),
    os.path.join(RESULTS_DIR, "sf_roi_risk_log.csv")
]
print("\n[INFO] S3 결과 아카이빙 진행 중...")
s3_backup_url = archive_local_csv_to_zip(csv_files, "sensor_fusion_results.zip")
print(f"[SUCCESS] S3 백업 아카이브 URL: {s3_backup_url}")

