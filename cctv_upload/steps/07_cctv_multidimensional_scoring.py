import os
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import cv2
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
from scipy import ndimage
import json
import re

# 우리가 작성한 기상청 API 연동 모듈 임포트
from weather_api import get_mangwon_weather

# =========================================================================
# 1. 경로 및 파일 설정
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MANGWON_IMAGE_DIR = r"E:\test\cctv_망원시장"
CCTV_BEV_CSV = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "mangwon_label_pedestrians.csv"))

summary_csv_path = os.path.join(RESULTS_DIR, "cctv_multidimensional_summary.csv")
history_csv_path = os.path.join(RESULTS_DIR, "cctv_pedestrian_history.csv")
roi_csv_path = os.path.join(RESULTS_DIR, "cctv_roi_multidimensional_log.csv")

# =========================================================================
# 2. 데이터 로딩 및 망원시장 sequence 11 정제
# =========================================================================
df = pd.read_csv(CCTV_BEV_CSV)
if 'sequence_id' in df.columns:
    cctv_df = df[df['sequence_id'] == 11].copy()
    print(f"[INFO] sequence_id = 11 필터링 완료! (데이터 수: {len(cctv_df)})")
else:
    cctv_df = df.copy()
    print(f"[INFO] sequence_id 컬럼이 없어 전체 데이터 로드 완료! (데이터 수: {len(cctv_df)})")

# 컬럼 맵핑 표준화 (frame_id -> frame 변환)
if 'frame_id' in cctv_df.columns:
    cctv_df.rename(columns={'frame_id': 'frame'}, inplace=True)

total_frames = int(cctv_df['frame'].max())
print(f"[INFO] 분석 대상 프레임: 1 ~ {total_frames}프레임")

# =========================================================================
# 3. 가우시안 히트맵 생성 헬퍼 함수
# =========================================================================
def create_cctv_gaussian_heatmap(c_sub, nx, ny, x_range, y_range, grid_size, sigma=0.8):
    heatmap = np.zeros((ny, nx))
    if len(c_sub) == 0:
        return heatmap
    
    x_coords = c_sub['bev_x_m'].values
    y_coords = c_sub['bev_y_m'].values
    
    x_idx = ((x_coords - x_range[0]) / grid_size).astype(int)
    y_idx = ((y_coords - y_range[0]) / grid_size).astype(int)
    
    valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    
    for cx, cy in zip(x_idx, y_idx):
        heatmap[cy, cx] += 1.0
        
    sigma_grid = sigma / grid_size
    heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma_grid)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

# =========================================================================
# 4. ROI 구역 다각형 및 격자 마스크 정의 (사다리꼴 50㎡ 구역)
# =========================================================================
roi_zones = [
    {
        "roi_id": 1,
        "roi_name": "Mangwon_Alley_Zone",
        "polygon": Polygon([(-4.5, 0.0), (-1.2, 0.0), (3.0, 10.0), (-4.5, 10.0)]),
        "capacity_limit": 17
    }
]

# 격자 공간 설정
nx, ny = 110, 80
x_range, y_range = (-8.0, 14.0), (-3.0, 13.0)
grid_size = 0.2

# 격자 마스크 생성
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

# 호모그래피 역행렬 계산 (BEV 2D 미터 -> 2D 픽셀 역투영)
H_MATRIX = np.array([
    [ 0.015, -0.002, -8.50],
    [ 0.001,  0.022, -2.10],
    [ 0.000,  0.001,  1.00]
])
H_INV = np.linalg.inv(H_MATRIX)

def transform_bev_to_pixel(bx, by):
    bev_pt = np.array([bx, by, 1.0])
    pixel_pt = H_INV.dot(bev_pt)
    pixel_pt /= pixel_pt[2]
    return int(pixel_pt[0]), int(pixel_pt[1])

# =========================================================================
# 5. 루프 실행 변수 정의 및 실행
# =========================================================================
all_weather_summary = {}

# OpenCV GUI 제어 창 및 실시간 트랙바 구성 (GUI 모드 차단 시 기본값 사용)
win_name = "Smart CCTV Multi-Dimensional Scoring Control"
try:
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.createTrackbar("Evac Threshold", win_name, 70, 100, lambda x: None)
    cv2.createTrackbar("Alarm Delay (s)", win_name, 3, 10, lambda x: None)
    cv2.createTrackbar("Density Weight (%)", win_name, 40, 100, lambda x: None)
except Exception:
    pass

for WEATHER_MODE in ["SUNNY", "RAINY", "HOT_SUMMER"]:
    print(f"\n============================================================")
    print(f"[WEATHER MODE] 현재 실행 중인 날씨 모드: {WEATHER_MODE}")
    print(f"============================================================")

    # 기상청 정보 사전 수집 및 환경 오프셋 설정
    weather_data = get_mangwon_weather(fallback_mode=WEATHER_MODE)
    IS_RAINY = weather_data.get('is_rainy', False)
    DISCOMFORT_INDEX = weather_data.get('discomfort_index', 75.0)

    # (1) 강수 상태에 따른 보행자 물리적 안전 임계거리 조율 (우산 보정 효과)
    SAFE_DISTANCE_THRESHOLD = 0.6 if IS_RAINY else 0.3

    # (2) 기상 취약점 점수 (S_weather) 정량화 (맑음=10, 폭염=40, 폭우=80)
    if IS_RAINY:
        S_WEATHER_SCORE = 80.0
    elif DISCOMFORT_INDEX >= 80.0:
        S_WEATHER_SCORE = 40.0
    else:
        S_WEATHER_SCORE = 10.0

    print(f"[WEATHER INFO] 기온: {weather_data.get('temp')}C, 습도: {weather_data.get('humidity')}%, 불쾌지수: {DISCOMFORT_INDEX}")
    print(f"[WEATHER FILTER] 안전거리 임계치: {SAFE_DISTANCE_THRESHOLD}m, 기상 취약점 점수: {S_WEATHER_SCORE}pt\n")

    # 모드별 비디오 및 결과 저장 경로 정의
    video_path = os.path.join(RESULTS_DIR, f"cctv_spatial_analysis_movie_{WEATHER_MODE.lower()}.mp4")
    summary_csv_path_mode = os.path.join(RESULTS_DIR, f"cctv_multidimensional_summary_{WEATHER_MODE.lower()}.csv")
    history_csv_path_mode = os.path.join(RESULTS_DIR, f"cctv_pedestrian_history_{WEATHER_MODE.lower()}.csv")
    roi_csv_path_mode = os.path.join(RESULTS_DIR, f"cctv_roi_multidimensional_log_{WEATHER_MODE.lower()}.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_width, video_height = 1920, 1080
    video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (video_width, video_height))

    summary_data = []
    pedestrian_history = []
    roi_risk_logs = []
    zone_risk_histories = {}
    last_valid_cctv_img = None
    pedestrian_tracks = {}
    evacuation_frames_counter = 0 # 대피 기준 초과 프레임 카운터 초기화

    print(f"[START] [{WEATHER_MODE}] 다차원 스코어링 및 기상 융합 분석 시작...")

    for frame_idx in range(1, total_frames + 1):
        c_sub = cctv_df[cctv_df['frame'] == frame_idx]
        cctv_count = len(c_sub)
        
        # 실시간 제어 바(트랙바) 설정 정보 획득
        evac_threshold = cv2.getTrackbarPos("Evac Threshold", win_name)
        alarm_delay_sec = cv2.getTrackbarPos("Alarm Delay (s)", win_name)
        w_density_percent = cv2.getTrackbarPos("Density Weight (%)", win_name)
        
        # 가중치 분배 연산 (밀집 가중치 w_density_percent를 조절하면 다른 가중치들이 비율에 맞게 변함)
        w_density = w_density_percent / 100.0
        w_remain = 1.0 - w_density
        w_count = w_remain * (2.0 / 6.0)
        w_stagnation = w_remain * (3.0 / 6.0)
        w_weather = w_remain * (1.0 / 6.0)
        
        # 가우시안 히트맵 생성
        cctv_heatmap = create_cctv_gaussian_heatmap(c_sub, nx, ny, x_range, y_range, grid_size)
        
        # 보행자 리스트 수집
        frame_pedestrians = []
        for _, row in c_sub.iterrows():
            pid = int(row['person_id'])
            cx = float(row['bev_x_m'])
            cy = float(row['bev_y_m'])
            
            # 픽셀 좌표 역변환
            if 'pixel_x' in row and not pd.isna(row['pixel_x']):
                px = int(row['pixel_x'])
                py = int(row['pixel_y'])
            else:
                px, py = transform_bev_to_pixel(cx, cy)
                
            # 궤적 업데이트 및 실시간 속도 산출
            if pid not in pedestrian_tracks:
                pedestrian_tracks[pid] = []
            pedestrian_tracks[pid].append((frame_idx, cx, cy))
            if len(pedestrian_tracks[pid]) > 5:
                pedestrian_tracks[pid].pop(0)
                
            # 보행 속도 계산 (10 FPS 기준)
            speed = 1.0  # 기본 속도값 (1.0m/s)
            if len(pedestrian_tracks[pid]) >= 2:
                f0, x0, y0 = pedestrian_tracks[pid][-2]
                f1, x1, y1 = pedestrian_tracks[pid][-1]
                dt = (f1 - f0) / 10.0 # 초 단위 변위 시간
                if dt > 0:
                    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                    speed = dist / dt
                    
            obj_type = "Pedestrian"
                
            frame_pedestrians.append({
                'pid': pid,
                'px': px,
                'py': py,
                'cx': cx,
                'cy': cy,
                'speed': speed,
                'obj_type': obj_type
            })
            
            # 개별 보행자 히스토리 백업
            pedestrian_history.append({
                'frame_id': frame_idx,
                'person_id': pid,
                'x_m': cx,
                'y_m': cy,
                'pixel_x': px,
                'pixel_y': py,
                'risk_score': 0.0 # 하단에서 거리 연산 후 채움
            })

        # 보행자 간 거리 행렬 연산
        close_mask = np.zeros(len(frame_pedestrians), dtype=bool)
        if len(frame_pedestrians) >= 2:
            coords_array = np.array([[p['cx'], p['cy']] for p in frame_pedestrians])
            dist_matrix = cdist(coords_array, coords_array)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # 1차 보정: 강수량 발생 시 안전거리 기준을 0.3m -> 0.6m로 2배 확장 적용 (우산 간 간격)
            close_mask = np.any(dist_matrix < SAFE_DISTANCE_THRESHOLD, axis=1)
            


        # 개별 보행자 위험도 정보 업데이트
        for idx, p in enumerate(frame_pedestrians):
            ind_risk = 5.0 if close_mask[idx] else 1.0

            
            for hist in reversed(pedestrian_history):
                if hist['frame_id'] == frame_idx and hist['person_id'] == p['pid']:
                    hist['risk_score'] = ind_risk
                    break

        # ROI 구역별 다차원 위험도 스코어링 산출
        max_risk_score = 0.0
        in_zone_count = 0
        
        s_count = 0.0
        s_density = 0.0
        s_stagnation = 0.0
        
        for zone in roi_zones:
            zone_count = 0
            zone_proximity_count = 0
            speeds_in_zone = []
            
            for idx, p in enumerate(frame_pedestrians):
                if zone["polygon"].contains(Point(p['cx'], p['cy'])):
                    zone_count += 1
                    if close_mask[idx]:
                        zone_proximity_count += 1
                    speeds_in_zone.append(p['speed'])
                        
            in_zone_count = max(in_zone_count, zone_count)
            
            # 1) 인원 규모 지수 (S_count) 연산
            s_count = min(((zone_count / float(zone["capacity_limit"])) * 100.0), 100.0)
            
            # 2) 대인 밀집 지수 (S_density) 연산
            mask = zone["mask"]
            avg_density = float(np.mean(cctv_heatmap[mask])) if np.sum(mask) > 0 else 0.0
            
            prox_ratio = (zone_proximity_count / float(zone_count)) * 50.0 if zone_count > 0 else 0.0
            dens_ratio = min((avg_density / 0.5), 1.0) * 50.0
            s_density = min(prox_ratio + dens_ratio, 100.0)
            
            # 3) 이동 지연 지수 (S_stagnation) 연산 (현실적인 보행 정체 보정)
            avg_speed = np.mean(speeds_in_zone) if speeds_in_zone else 1.0
            # 속도가 0.8m/s 이상이면 정상 이동(0pt), 0.2m/s 이하로 정체될 때 지연 점수 증가
            if avg_speed >= 0.8:
                s_stagnation = 0.0
            else:
                s_stagnation = max(0.0, min(100.0, ((0.8 - avg_speed) / 0.6) * 100.0))
            
            # 4) 종합 위험 지수 (CRI) 연산 (실시간 트랙바 가중치 반영)
            cri_raw = (w_count * s_count) + (w_density * s_density) + (w_stagnation * s_stagnation) + (w_weather * S_WEATHER_SCORE)
            
            cri_score = min(float(cri_raw), 100.0)
            
            if zone["roi_id"] not in zone_risk_histories:
                zone_risk_histories[zone["roi_id"]] = []
            zone_risk_histories[zone["roi_id"]].append(cri_score)
            if len(zone_risk_histories[zone["roi_id"]]) > 10:
                zone_risk_histories[zone["roi_id"]].pop(0)
                
            smoothed_cri = float(np.mean(zone_risk_histories[zone["roi_id"]]))
            max_risk_score = max(max_risk_score, smoothed_cri)
            
            # 실시간 트랙바 임계값 기준으로 대피 / 혼잡 판단
            risk_level = "정상"
            if smoothed_cri >= evac_threshold:
                risk_level = "대피"
            elif smoothed_cri >= (evac_threshold * 3.0 / 7.0): # 70대비 30의 비율
                risk_level = "혼잡"
                
            roi_risk_logs.append({
                'frame_id': frame_idx,
                'roi_id': zone["roi_id"],
                'roi_name': zone["roi_name"],
                'pedestrian_count': zone_count,
                'average_density': avg_density,
                's_count': round(s_count, 1),
                's_density': round(s_density, 1),
                's_stagnation': round(s_stagnation, 1),
                's_weather': S_WEATHER_SCORE,
                'risk_score': round(smoothed_cri, 1),
                'risk_level': risk_level
            })
            
        summary_data.append({
            'frame_id': frame_idx,
            'timestamp_sec': float(frame_idx / 10.0),
            'cctv_count': in_zone_count,
            'lidar_added_count': 0,
            'total_fusion_count': cctv_count,
            's_count': round(s_count, 1),
            's_density': round(s_density, 1),
            's_stagnation': round(s_stagnation, 1),
            's_weather': S_WEATHER_SCORE,
            'max_risk_score': round(max_risk_score, 1)
        })

        # 망원시장 CCTV 이미지 로드 (순환 인덱싱으로 604프레임 전체 누락 방지)
        mangwon_files = sorted([f for f in os.listdir(MANGWON_IMAGE_DIR) if f.endswith('.jpg')])
        cctv_color = None
        if mangwon_files:
            target_file = mangwon_files[(frame_idx - 1) % len(mangwon_files)]
            img_path = os.path.join(MANGWON_IMAGE_DIR, target_file)
            try:
                img_array = np.fromfile(img_path, np.uint8)
                cctv_color = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if cctv_color is not None:
                    last_valid_cctv_img = cctv_color.copy()
            except Exception:
                cctv_color = None

        if cctv_color is None and last_valid_cctv_img is not None:
            cctv_color = last_valid_cctv_img.copy()

        if cctv_color is None:
            cctv_color = np.zeros((video_height, video_width, 3), dtype=np.uint8)

        # 1) ROI 사다리꼴 경계선 그리기
        pts = np.array([transform_bev_to_pixel(x, y) for x, y in roi_zones[0]["polygon"].exterior.coords], dtype=np.int32)
        cv2.polylines(cctv_color, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

        # 2) 보행자 개별 렌더링
        for idx, p in enumerate(frame_pedestrians):
            if not roi_zones[0]["polygon"].contains(Point(p['cx'], p['cy'])):
                continue
                
            dot_color = (0, 0, 255) if close_mask[idx] else (0, 255, 0)
            label_text = f"P (ID:{p['pid']})"

            cv2.circle(cctv_color, (p['px'], p['py']), radius=7, color=dot_color, thickness=-1)
            cv2.circle(cctv_color, (p['px'], p['py']), radius=7, color=(255, 255, 255), thickness=1)
            
            cv2.putText(cctv_color, label_text, (p['px'] - 30, p['py'] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(cctv_color, label_text, (p['px'] - 30, p['py'] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)

        # 3) 밀착 관계선 그리기
        if len(frame_pedestrians) >= 2:
            for i in range(len(frame_pedestrians)):
                for j in range(i + 1, len(frame_pedestrians)):
                    p1, p2 = frame_pedestrians[i], frame_pedestrians[j]
                    if roi_zones[0]["polygon"].contains(Point(p1['cx'], p1['cy'])) and roi_zones[0]["polygon"].contains(Point(p2['cx'], p2['cy'])):
                        dist = dist_matrix[i, j]
                        limit = SAFE_DISTANCE_THRESHOLD

                            
                        if dist < limit:
                            cv2.line(cctv_color, (p1['px'], p1['py']), (p2['px'], p2['py']), (0, 0, 255), 1)

        # 4) ROI 센트로이드 텍스트
        zone_log = next((log for log in reversed(roi_risk_logs) if log['roi_id'] == roi_zones[0]['roi_id'] and log['frame_id'] == frame_idx), None)
        if zone_log:
            centroid = roi_zones[0]["polygon"].centroid
            rx_img, ry_img = transform_bev_to_pixel(centroid.x, centroid.y)
            
            level = zone_log["risk_level"]
            t_color = (0, 0, 255) if level == "대피" else (0, 255, 255) if level == "혼잡" else (0, 255, 0)
            
            text_str = f"Alley: {zone_log['risk_score']:.1f}pt ({zone_log['pedestrian_count']}p)"
            cv2.putText(cctv_color, text_str, (rx_img - 50, ry_img + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(cctv_color, text_str, (rx_img - 50, ry_img), cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1)

        # 대피 수준 지속 시간 체크 및 출동 알람 판단 (10 FPS 기준)
        required_alarm_frames = max(1, alarm_delay_sec * 10)
        if max_risk_score >= evac_threshold:
            evacuation_frames_counter += 1
        else:
            evacuation_frames_counter = 0
            
        dispatch_alarm_active = (evacuation_frames_counter >= required_alarm_frames)

        # 5) 실시간 HUD
        state_label = "NORMAL / 정상"
        hud_color = (0, 255, 0)
        if max_risk_score >= evac_threshold:
            state_label = "EVACUATE / 대피"
            hud_color = (0, 0, 255)
        elif max_risk_score >= (evac_threshold * 3.0 / 7.0):
            state_label = "CONGESTED / 혼잡"
            hud_color = (0, 255, 255)

        # 대피 수준 지속 알람 정보 출력
        alarm_label = "DISPATCH: SAFE"
        if dispatch_alarm_active:
            # 빨간색과 주황색으로 깜빡이는 효과 (프레임 인덱스 이용)
            flash_color = (0, 0, 255) if (frame_idx % 6 < 3) else (0, 140, 255)
            alarm_label = "🚨 DISPATCH ALARM! 🚨"
            
            # 화면 상단 중앙에 크게 경보 메시지 오버레이
            cv2.rectangle(cctv_color, (500, 30), (1350, 100), (0, 0, 0), -1)
            cv2.rectangle(cctv_color, (500, 30), (1350, 100), flash_color, 3)
            cv2.putText(cctv_color, f"EMERGENCY ALARM: DISPATCH ACTIVE ({alarm_delay_sec}s Maintained)", (520, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, flash_color, 2)
        elif evacuation_frames_counter > 0:
            current_held_sec = evacuation_frames_counter / 10.0
            alarm_label = f"EVAC HELD: {current_held_sec:.1f}s / {alarm_delay_sec}s"

        overlay = cctv_color.copy()
        hud_x1, hud_y1, hud_x2, hud_y2 = 1420, 30, 1890, 230
        cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), hud_color, -1)
        cv2.addWeighted(overlay, 0.4, cctv_color, 0.6, 0, cctv_color)
        
        cv2.rectangle(cctv_color, (hud_x1, hud_y1), (hud_x2, hud_y2), (255, 255, 255), 2)
        
        weather_emoji = "🌧️" if IS_RAINY else "🥵" if DISCOMFORT_INDEX >= 80.0 else "☀️"
        weather_label = "폭우/우산보정" if IS_RAINY else "폭염/불쾌지수높음" if DISCOMFORT_INDEX >= 80.0 else "맑음/쾌적"

        cv2.putText(cctv_color, f"STATUS: {state_label}", (hud_x1 + 15, hud_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(cctv_color, f"CRI SCORE: {max_risk_score:.1f}pt", (hud_x1 + 15, hud_y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(cctv_color, f"ZONE PEOPLES: {in_zone_count}p", (hud_x1 + 15, hud_y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(cctv_color, f"WEATHER: {weather_label} ({weather_emoji})", (hud_x1 + 15, hud_y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # 실시간 가중치 및 대피 임계값 시각화
        cv2.putText(cctv_color, f"Evac Limit: {evac_threshold}pt", (hud_x1 + 15, hud_y1 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(cctv_color, f"Density Wt: {w_density_percent}%", (hud_x1 + 15, hud_y1 + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(cctv_color, alarm_label, (hud_x1 + 15, hud_y1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if dispatch_alarm_active else (255, 255, 255), 2)

        cv2.putText(cctv_color, f"MANGWON MARKET SMART CCTV - Frame {frame_idx}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cctv_color, f"ZONE COUNT: {in_zone_count}p", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        video_writer.write(cctv_color)

        # 실시간 제어화면 출력 (GUI 환경 지원)
        try:
            cv2.imshow(win_name, cctv_color)
            cv2.waitKey(1)
        except Exception:
            pass

        # [모드별 프레임 폴더로 이미지 저장]
        frames_dir = os.path.join(RESULTS_DIR, f"frames_{WEATHER_MODE.lower()}")
        os.makedirs(frames_dir, exist_ok=True)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:03d}.jpg"), cctv_color)

        # 기본 frames 디렉터리 동기화 복사
        default_frames_dir = os.path.join(RESULTS_DIR, "frames")
        os.makedirs(default_frames_dir, exist_ok=True)
        cv2.imwrite(os.path.join(default_frames_dir, f"frame_{frame_idx:03d}.jpg"), cctv_color)

    video_writer.release()
    print(f"[SUCCESS] {WEATHER_MODE} 다차원 관제 비디오 생성 완료!")

    # 모드별 결과 백업 CSV 작성
    pd.DataFrame(summary_data).to_csv(summary_csv_path_mode, index=False)
    pd.DataFrame(pedestrian_history).to_csv(history_csv_path_mode, index=False)
    pd.DataFrame(roi_risk_logs).to_csv(roi_csv_path_mode, index=False)
    print(f"[SUCCESS] {WEATHER_MODE} CSV 결과 저장 완료!")

    # 요약 데이터 누적
    all_weather_summary[WEATHER_MODE] = summary_data

# =========================================================================
# 6. 웹 대시보드 컴파일 및 분석 데이터 주입
# =========================================================================
dashboard_html_path = r"E:\AIVLE_10team\results\cctv_simulation_dashboard.html"
if os.path.exists(dashboard_html_path):
    with open(dashboard_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    json_data = json.dumps(all_weather_summary, ensure_ascii=False)
    
    # 샌드위치 마커 사이를 통째로 치환하는 가장 안전하고 명확한 방식
    pattern = r"// === WEATHER_DATA_START ===[\s\S]*?// === WEATHER_DATA_END ==="
    replacement = f"// === WEATHER_DATA_START ===\n        const cctvSummaryData = {json_data};\n        // === WEATHER_DATA_END ==="
    updated_html = re.sub(pattern, replacement, html_content)
    
    with open(dashboard_html_path, "w", encoding="utf-8") as f:
        f.write(updated_html)
    print("[SUCCESS] cctv_simulation_dashboard.html 데이터 동기화 완료!")

# 모든 작업 종료 후 OpenCV 창 닫기
cv2.destroyAllWindows()
print("============================================================\n")
