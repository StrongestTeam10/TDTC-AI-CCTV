# 06_cctv_spatial_analysis.py
# ===================================================================================================
# [목적 및 역할]
# 본 스크립트는 CCTV(카메라) BEV 역투영 밀도 데이터를 기반으로 
# LiDAR 융합 없이 CCTV 단독으로 공간 분석(ROI 위험도 및 인원 카운팅)을 수행하고
# 분석 결과를 백엔드(Supabase DB 및 AWS S3)로 자동 적재 및 시각화 비디오를 생성하는 스크립트입니다.
# ===================================================================================================

import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
# pyrefly: ignore [missing-import]
import cv2
from shapely.geometry import Polygon, Point

# 백업된 sensor_fusion_archive 폴더 하위의 utils 모듈 참조를 위해 sys.path에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "sensor_fusion_archive"))

try:
    # pyrefly: ignore [missing-import]
    from utils.db_connector import bulk_insert_summary, bulk_insert_history, bulk_insert_roi_log, bulk_insert_risk_score, bulk_insert_emergency_alert
    # pyrefly: ignore [missing-import]
    from utils.s3_uploader import archive_local_csv_to_zip
except ImportError:
    print("[WARNING] 백엔드 연동 utils 모듈을 로드하지 못했습니다. 시뮬레이션 모드로 작동합니다.")
    bulk_insert_summary = None
    bulk_insert_history = None
    bulk_insert_roi_log = None
    bulk_insert_risk_score = None
    bulk_insert_emergency_alert = None
    archive_local_csv_to_zip = None

# =========================================================================
# 1. 경로 및 하이퍼파라미터 설정
# =========================================================================
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1) 입력 좌표 데이터 경로
CCTV_BEV_CSV = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "cctv_bev_coordinates_mall.csv"))

# 2) 출력 데이터 경로
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "cctv_spatial_summary.csv")
HISTORY_CSV_PATH = os.path.join(RESULTS_DIR, "cctv_pedestrian_history.csv")
ROI_LOG_CSV_PATH = os.path.join(RESULTS_DIR, "cctv_roi_risk_log.csv")
ZIP_ARCHIVE_PATH = os.path.join(RESULTS_DIR, "cctv_spatial_results.zip")

# =========================================================================
# 2. CCTV BEV 좌표 파일 로드 및 총 프레임 계산
# =========================================================================
if not os.path.exists(CCTV_BEV_CSV):
    print(f"[ERROR] CCTV BEV CSV 파일을 찾을 수 없습니다: {CCTV_BEV_CSV}")
    sys.exit(1)

cctv_df = pd.read_csv(CCTV_BEV_CSV)
if len(cctv_df) == 0:
    total_frames = 100 # 기본 프레임수
else:
    total_frames = int(cctv_df['frame'].max())

print(f"[INFO] CCTV BEV CSV 로딩 완료! (총 프레임: {total_frames}, 보행자 감지 행수: {len(cctv_df)})")

# =========================================================================
# 3. 가우시안 BEV 히트맵 매핑 헬퍼 함수
# =========================================================================
def create_cctv_gaussian_heatmap(df_frame, nx, ny, x_range, y_range, grid_size, sigma_m=0.8):
    heatmap = np.zeros((ny, nx), dtype=np.float32)
    sigma_grid = sigma_m / grid_size

    for _, row in df_frame.iterrows():
        cx = row['bev_x_m']
        cy = row['bev_y_m']
        
        # 물리 좌표 -> 격자 좌표 변환
        gx = (cx - x_range[0]) / grid_size
        gy = (cy - y_range[0]) / grid_size
        
        # 격자 범위 필터링
        if 0 <= gx < nx and 0 <= gy < ny:
            # 7x7 범위 내 로컬 가우시안 맵 적용
            kernel_radius = int(3 * sigma_grid)
            for dy_in in range(-kernel_radius, kernel_radius + 1):
                for dx_in in range(-kernel_radius, kernel_radius + 1):
                    x_target = int(gx + dx_in)
                    y_target = int(gy + dy_in)
                    if 0 <= x_target < nx and 0 <= y_target < ny:
                        dist_sq = (dx_in * grid_size)**2 + (dy_in * grid_size)**2
                        weight = np.exp(-dist_sq / (2 * (sigma_m**2)))
                        heatmap[y_target, x_target] += weight
                        
    return np.clip(heatmap, 0.0, 1.0)

# =========================================================================
# 4. 히트맵 로컬 피크 검출을 통한 보행자 3D 물리 좌표 획득 함수
# =========================================================================
def extract_peak_coordinates(heatmap, nx, ny, x_range, y_range, grid_size, threshold=0.15):
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
# 5. ROI 구역 다각형 및 정원 제한 정의
# =========================================================================
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

# 성능 가속화: ROI 격자 마스크 사전 계산
for zone in roi_zones:
    poly = zone["polygon"]
    mask = np.zeros((ny, nx), dtype=bool)
    for y_idx in range(ny):
        for x_idx in range(nx):
            cx = x_range[0] + x_idx * grid_size + 0.5 * grid_size
            cy = y_range[0] + y_idx * grid_size + 0.5 * grid_size
            pt = Point(cx, cy)
            if poly.contains(pt):
                mask[y_idx, x_idx] = True
    zone["mask"] = mask

# =========================================================================
# 6. 프레임별 공간 분석 및 시각화 비디오 쓰기
# =========================================================================
summary_data = []
roi_risk_logs = []
pedestrian_history = []

# 비디오 렌더링 파일명 지정 ( results/cctv_spatial_analysis_movie.mp4 )
video_path = os.path.join(RESULTS_DIR, "cctv_spatial_analysis_movie.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 해상도: 가로 360(nx * 3), 세로 480(ny * 3)
video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (nx * 3, ny * 3))

print("[START] [CCTV 단독 공간 분석 시작] 총 프레임 분석 중...")

for frame_idx in range(1, total_frames + 1):
    c_sub = cctv_df[cctv_df['frame'] == frame_idx]
    cctv_count = len(c_sub)
    
    # CCTV BEV 가우시안 히트맵 생성
    cctv_heatmap = create_cctv_gaussian_heatmap(c_sub, nx, ny, x_range, y_range, grid_size)
    
    # 프레임 내 실제 보행자 좌표 추출
    frame_coords = list(zip(c_sub['bev_x_m'].values, c_sub['bev_y_m'].values))
    cctv_detected_count = len(frame_coords)
    
    # 보행자 좌표 적재 데이터 축적 (pedcord01h 스키마 규격)
    for p_idx, (cx, cy) in enumerate(frame_coords, start=1):
        pedestrian_history.append({
            'zone_id': 1,
            'frame_id': frame_idx,
            'person_id': p_idx,
            'pixel_x': float(cx * 10.0),
            'pixel_y': float(cy * 10.0),
            'bev_x_m': float(cx),
            'bev_y_m': float(cy),
            'risk_score': 0.0,
            'analysis_mode': 'LIVE',
            'video_id': 1
        })

    # ROI 위험도 및 밀집도 산출 (crddnst01m / crddnst01h 스키마 규격)
    max_risk_score = 0.0
    for zone in roi_zones:
        # ROI 내 속한 보행자 수 계산 (실제 보행자 BEV 좌표 기준)
        zone_count = sum(1 for cx, cy in frame_coords if zone["polygon"].contains(Point(cx, cy)))
        
        # 격자 마스크 영역의 평균 점유 강도
        mask = zone["mask"]
        avg_density = float(np.mean(cctv_heatmap[mask])) if np.sum(mask) > 0 else 0.0
        
        # 위험도 스코어 산출 공식
        risk_score = (15.0 * zone_count) + (50.0 * avg_density)
        max_risk_score = max(max_risk_score, risk_score)
        
        # 위험 등급 판정
        risk_level = "정상"
        if risk_score >= 100.0:
            risk_level = "심각"
        elif risk_score >= 50.0:
            risk_level = "경고"
        elif risk_score >= 20.0:
            risk_level = "주의"
            
        roi_risk_logs.append({
            'zone_id': zone["roi_id"],
            'visitor_count': zone_count,
            'density_score': avg_density,
            'status_level': risk_level,
            'analysis_mode': 'LIVE',
            'video_id': 1,
            'frame_id': frame_idx
        })
        
    # 시장 요약 데이터 축적 (mktsmry01s 스키마 규격)
    summary_data.append({
        'market_id': 1,
        'frame_id': frame_idx,
        'total_cctv_count': cctv_count,
        'avg_density_score': float(cctv_heatmap.mean()),
        'max_density_score': float(cctv_heatmap.max()),
        'max_risk_score': max_risk_score,
        'analysis_mode': 'LIVE',
        'video_id': 1
    })

    # 시각화 이미지 빌드
    cctv_scaled = (np.clip(cctv_heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    cctv_color = cv2.applyColorMap(cctv_scaled, cv2.COLORMAP_JET)
    
    # ROI 다각형 경계 그리기
    for zone in roi_zones:
        poly = zone["polygon"]
        pts = np.array([[(x - x_range[0]) / grid_size, (y - y_range[0]) / grid_size] for x, y in poly.exterior.coords], dtype=np.int32)
        cv2.polylines(cctv_color, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
        
    cctv_color_resized = cv2.resize(cctv_color, (nx * 3, ny * 3))
    cv2.putText(cctv_color_resized, f"CCTV BEV Heatmap - Frame {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 비디오에 작성
    video_writer.write(cctv_color_resized)

# 비디오 닫기
video_writer.release()
print(f"[SUCCESS] CCTV 공간 분석 비디오 저장 완료: {video_path}")

# 하위 호환성을 위해 루트 results/ 폴더에도 비디오 및 결과 복사
try:
    root_results = os.path.join(os.path.dirname(BASE_DIR), "results")
    os.makedirs(root_results, exist_ok=True)
    import shutil
    shutil.copy(video_path, os.path.join(root_results, os.path.basename(video_path)))
except Exception:
    pass

# =========================================================================
# 7. 로컬 백업 CSV 저장
# =========================================================================
df_summary = pd.DataFrame(summary_data)
df_history = pd.DataFrame(pedestrian_history)
df_roi = pd.DataFrame(roi_risk_logs)

df_summary.to_csv(SUMMARY_CSV_PATH, index=False)
df_history.to_csv(HISTORY_CSV_PATH, index=False)
df_roi.to_csv(ROI_LOG_CSV_PATH, index=False)

print("============================================================")
print("[COMPLETED] [CCTV 단독 공간 분석 및 로컬 CSV 저장 완료]")
print("============================================================")
print(f"[FILE] 프레임 요약 저장: {SUMMARY_CSV_PATH}")
print(f"[FILE] 보행자 좌표 저장: {HISTORY_CSV_PATH}")
print(f"[FILE] ROI 위험 로그 저장: {ROI_LOG_CSV_PATH}")
print(f"[AVG_DETECT] 프레임당 CCTV 평균 탐지: {df_summary['total_cctv_count'].mean():.1f} 명")
print(f"[MAX_RISK] 최고 위험 영역 위험도 평균: {df_summary['max_risk_score'].mean():.2f} 점")
print("============================================================")

# =========================================================================
# 8. 클라우드 백엔드 (Supabase & S3) 적재
# =========================================================================
print("\n[INFO] 클라우드 백엔드 데이터 적재 시작...")

# 1) Supabase DB Bulk Insert
if bulk_insert_summary is not None:
    bulk_insert_summary(df_summary.to_dict('records'))
    bulk_insert_history(df_history.to_dict('records'))
    bulk_insert_roi_log(df_roi.to_dict('records'))
    
    # 위험 점수 (mrkrisk01m) 및 긴급 알림 (emgalrt01h) 데이터 생성 및 적재
    high_risk_list = [r for r in df_roi.to_dict('records') if r.get('density_score', 0) > 0.01]
    if high_risk_list and bulk_insert_risk_score is not None:
        risk_records = [{
            'zone_id': r.get('zone_id', 1),
            'risk_score': float(r.get('density_score', 0.0) * 100),
            'risk_level': r.get('status_level', 'NORMAL'),
            'reason_code': 'CROWD_DENSITY_HIGH',
            'frame_id': r.get('frame_id', 1)
        } for r in high_risk_list]
        bulk_insert_risk_score(risk_records)

        alert_records = [{
            'zone_id': r.get('zone_id', 1),
            'risk_id': 1,
            'alert_type': 'CROWD_SURGE_ALERT',
            'is_resolved': False
        } for r in high_risk_list[:5]] # 상위 5건 긴급알림 적재
        bulk_insert_emergency_alert(alert_records)
else:
    print("[INFO] [DB 시뮬레이션] mktsmry01s / pedcord01h / crddnst01m / mrkrisk01m / emgalrt01h 적재 시뮬레이션 완료")

# 2) AWS S3 결과물 압축 백업 업로드
if archive_local_csv_to_zip is not None:
    # CCTV 전용 파일들을 백업 압축 업로드 대상으로 지정
    csv_paths = [
        os.path.join(RESULTS_DIR, "cctv_spatial_summary.csv"),
        os.path.join(RESULTS_DIR, "cctv_pedestrian_history.csv"),
        os.path.join(RESULTS_DIR, "cctv_roi_risk_log.csv")
    ]
    s3_url = archive_local_csv_to_zip(csv_paths, "cctv_spatial_results.zip")
    print(f"[SUCCESS] S3 백업 업로드 완료: {s3_url}")
else:
    print("[SUCCESS] S3 백업 아카이브 URL: https://mock-bucket.s3.amazonaws.com/backups/cctv_spatial_results.zip")
