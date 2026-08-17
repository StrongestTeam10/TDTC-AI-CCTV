# 09_aggregate_pedestrian_json.py
# ===================================================================================================
# [목적 및 역할]
# 갱신된 [new_table.txt] Supabase 적재 가이드 (최신 버전)에 100% 매칭
# 1) pedaggr01h 컬럼 규격: clip_id, frame_id, video_id, pixels_json, bev_xyz_json, captured_at
# 2) JSON 포맷: {"person_1": {"x": 359.0, "y": 37.0}} 또는 {"1": {"x": 359.0, "y": 37.0}}
# 3) ROI Zone 필터링 + 글로벌 객체 추적기(Global Track ID) + EMA 튐 보정 적용
# ===================================================================================================

import os
import sys
import json
import math
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. 입력 CSV 로드 (환경변수 INPUT_PEDESTRIAN_CSV를 최우선으로 적용)
env_input_csv = os.environ.get("INPUT_PEDESTRIAN_CSV")
input_csv = None

if env_input_csv and os.path.exists(env_input_csv):
    input_csv = env_input_csv
else:
    INPUT_CSV_CANDIDATES = [
        os.path.join(RESULTS_DIR, "mangwon_label_pedestrians.csv"),
        os.path.join(RESULTS_DIR, "cctv_pedestrian_history.csv")
    ]
    for cand in INPUT_CSV_CANDIDATES:
        if os.path.exists(cand):
            input_csv = cand
            break

if not input_csv:
    print(f"[ERROR] 입력 보행자 CSV 파일을 찾을 수 없습니다.")
    sys.exit(1)

print(f"[INFO] 보행자 좌표 입력 데이터 로드: {input_csv}")
df = pd.read_csv(input_csv)

if 'frame' in df.columns and 'frame_id' not in df.columns:
    df.rename(columns={'frame': 'frame_id'}, inplace=True)

if 'pixel_u' in df.columns and 'pixel_x' not in df.columns:
    df.rename(columns={'pixel_u': 'pixel_x'}, inplace=True)
if 'pixel_v' in df.columns and 'pixel_y' not in df.columns:
    df.rename(columns={'pixel_v': 'pixel_y'}, inplace=True)

if 'bev_z_m' not in df.columns:
    df['bev_z_m'] = 0.0

if 'sequence_id' in df.columns:
    unique_seqs = df['sequence_id'].unique()
    target_seq = 11 if 11 in unique_seqs else unique_seqs[0]
    df = df[df['sequence_id'] == target_seq].copy()

# =========================================================================
# 2. 사다리꼴 관제 구역 (ROI Zone 다각형) 필터링
# =========================================================================
zone_id = str(os.environ.get("ZONE_ID", "1"))

ROI_2D_POLY = None
try:
    config_path = os.path.join(os.path.dirname(BASE_DIR), "core", "zones_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, "core", "zones_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            zones_config = json.load(f)
            zone_meta = zones_config.get(zone_id, zones_config.get("1", {}))
            ROI_2D_POLY = np.array(zone_meta.get("roi_polygon"), dtype=np.int32)
            print(f"✅ [CONFIG] zones_config.json에서 Zone {zone_id} ROI 다각형 로드 완료!")
except Exception as e:
    print(f"⚠️ [CONFIG] zones_config.json 로드 중 오류 발생, 기본 ROI 사용: {e}")

if ROI_2D_POLY is None:
    ROI_2D_POLY = np.array([
        [70, 1000],    # 좌하단
        [320, 220],    # 좌상단
        [940, 220],    # 우상단
        [1120, 900]    # 우하단
    ], dtype=np.int32)

def is_inside_roi(row):
    if 'in_roi' in row and pd.notnull(row['in_roi']) and str(row['in_roi']).lower() in ['true', '1']:
        return True
    px = float(row['pixel_x']) if pd.notnull(row.get('pixel_x')) else -1.0
    py = float(row['pixel_y']) if pd.notnull(row.get('pixel_y')) else -1.0
    if px < 0 or py < 0:
        return False
    return cv2.pointPolygonTest(ROI_2D_POLY, (px, py), False) >= 0

df['valid_roi'] = df.apply(is_inside_roi, axis=1)
df_roi = df[df['valid_roi']].copy()

print(f"[INFO] 사다리꼴 ROI Zone 필터링 (원 데이터: {len(df)}행 -> ROI 적용: {len(df_roi)}행)")

# =========================================================================
# 3. 글로벌 객체 추적기 (Global Track ID & EMA Smoothing)
# =========================================================================
df_roi.sort_values(by='frame_id', inplace=True)
frame_groups = df_roi.groupby('frame_id')

active_tracks = {}
next_global_id = 1
MAX_DISAPPEARED = 3
MAX_MATCH_DIST_M = 1.8

tracked_records = []

for frame_id, f_df in frame_groups:
    curr_detections = []
    for _, row in f_df.iterrows():
        curr_detections.append({
            'pixel_x': float(row['pixel_x']) if pd.notnull(row.get('pixel_x')) else 0.0,
            'pixel_y': float(row['pixel_y']) if pd.notnull(row.get('pixel_y')) else 0.0,
            'bev_x_m': float(row['bev_x_m']) if pd.notnull(row.get('bev_x_m')) else 0.0,
            'bev_y_m': float(row['bev_y_m']) if pd.notnull(row.get('bev_y_m')) else 0.0,
            'bev_z_m': float(row.get('bev_z_m', 0.0))
        })
    
    if not active_tracks:
        for det in curr_detections:
            active_tracks[next_global_id] = {
                'bev_x': det['bev_x_m'], 'bev_y': det['bev_y_m'],
                'pixel_x': det['pixel_x'], 'pixel_y': det['pixel_y'],
                'bev_z': det['bev_z_m'],
                'disappeared': 0
            }
            det['global_id'] = next_global_id
            det['frame_id'] = frame_id
            tracked_records.append(det)
            next_global_id += 1
    else:
        active_ids = list(active_tracks.keys())
        active_coords = np.array([[active_tracks[gid]['bev_x'], active_tracks[gid]['bev_y']] for gid in active_ids])
        
        if len(curr_detections) > 0:
            curr_coords = np.array([[d['bev_x_m'], d['bev_y_m']] for d in curr_detections])
            dist_matrix = cdist(active_coords, curr_coords)
            
            rows = dist_matrix.min(axis=1).argsort()
            cols = dist_matrix.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dist_matrix[row, col] > MAX_MATCH_DIST_M:
                    continue
                
                gid = active_ids[row]
                det = curr_detections[col]
                
                alpha_bev, alpha_px = 0.45, 0.50
                smooth_bx = alpha_bev * det['bev_x_m'] + (1 - alpha_bev) * active_tracks[gid]['bev_x']
                smooth_by = alpha_bev * det['bev_y_m'] + (1 - alpha_bev) * active_tracks[gid]['bev_y']
                smooth_px = alpha_px * det['pixel_x'] + (1 - alpha_px) * active_tracks[gid]['pixel_x']
                smooth_py = alpha_px * det['pixel_y'] + (1 - alpha_px) * active_tracks[gid]['pixel_y']
                
                active_tracks[gid] = {
                    'bev_x': smooth_bx, 'bev_y': smooth_by,
                    'pixel_x': smooth_px, 'pixel_y': smooth_py,
                    'bev_z': det['bev_z_m'],
                    'disappeared': 0
                }
                
                det['bev_x_m'] = round(smooth_bx, 3)
                det['bev_y_m'] = round(smooth_by, 3)
                det['pixel_x'] = round(smooth_px, 2)
                det['pixel_y'] = round(smooth_py, 2)
                det['global_id'] = gid
                det['frame_id'] = frame_id
                tracked_records.append(det)
                
                used_rows.add(row)
                used_cols.add(col)
            
            for col_idx, det in enumerate(curr_detections):
                if col_idx not in used_cols:
                    active_tracks[next_global_id] = {
                        'bev_x': det['bev_x_m'], 'bev_y': det['bev_y_m'],
                        'pixel_x': det['pixel_x'], 'pixel_y': det['pixel_y'],
                        'bev_z': det['bev_z_m'],
                        'disappeared': 0
                    }
                    det['global_id'] = next_global_id
                    det['frame_id'] = frame_id
                    tracked_records.append(det)
                    next_global_id += 1
            
            for row_idx, gid in enumerate(active_ids):
                if row_idx not in used_rows:
                    active_tracks[gid]['disappeared'] += 1
                    if active_tracks[gid]['disappeared'] > MAX_DISAPPEARED:
                        del active_tracks[gid]

df_tracked = pd.DataFrame(tracked_records)
print(f"[INFO] 글로벌 추적 및 평활화 완료 (총 보행자 객체: {next_global_id - 1}명)")

# =========================================================================
# 4. 신규 new_table.txt 스키마 규격 매핑 및 데이터 생성
# =========================================================================
# 1. pedaggr01h 신규 스키마:
#    - clip_id (int8, NOT NULL, FK)
#    - frame_id (int4, NOT NULL)
#    - video_id (int4, NULL)
#    - pixels_json (jsonb) -> {"person_1": {"x": 1920, "y": 1080}} 또는 {"1": {"x": 1920.0, "y": 1080.0}}
#    - bev_xyz_json (jsonb) -> {"person_1": {"x": 12.5, "y": 5.2, "z": 0.0}}
#    - captured_at (timestamp, NOT NULL)

aggregated_records = []
pixels_by_frame_dict = {}
bev_xyz_by_frame_dict = {}
trajectories_by_id = {}

if df_tracked.empty:
    print("[WARNING] 글로벌 추적 결과 보행자 데이터가 전혀 없습니다. (ROI 영역 내 검출 없음)")
else:
    df_tracked.sort_values(by=['frame_id', 'global_id'], inplace=True)
    grouped = df_tracked.groupby('frame_id')

    # 환경변수 로드
    env_clip_id = int(os.environ.get("CLIP_ID", 1))
    env_zone_id = int(os.environ.get("ZONE_ID", 1))
    env_s3_url = os.environ.get("S3_CLIP_URL", "https://s3.amazonaws.com/mangwon/cctv_clip_01.mp4")

    env_start_time_str = os.environ.get("START_TIME")
    if env_start_time_str:
        try:
            # ISO 포맷 파싱 시도
            start_time = datetime.fromisoformat(env_start_time_str.replace("Z", "+00:00"))
        except ValueError:
            try:
                start_time = datetime.strptime(env_start_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start_time = datetime.now(timezone.utc)
    else:
        start_time = datetime.now(timezone.utc)

    env_fps = float(os.environ.get("FPS", 15.0))

    for frame_id, group in grouped:
        pixels_map = {}
        bev_xyz_map = {}
        
        for _, row in group.iterrows():
            pid_key = f"person_{int(row['global_id'])}"
            px = round(float(row['pixel_x']), 2)
            py = round(float(row['pixel_y']), 2)
            bx = round(float(row['bev_x_m']), 3)
            by = round(float(row['bev_y_m']), 3)
            bz = round(float(row['bev_z_m']), 3)
            
            # 신규 적재 가이드 객체 형태: {"x": px, "y": py}
            pixels_map[pid_key] = {"x": px, "y": py}
            bev_xyz_map[pid_key] = {"x": bx, "y": by, "z": bz}
        
        frame_key = str(int(frame_id))
        pixels_by_frame_dict[frame_key] = pixels_map
        bev_xyz_by_frame_dict[frame_key] = bev_xyz_map
        
        # 프레임에 따른 타임스탬프 계산
        frame_sec = int(frame_id) / env_fps
        frame_time = start_time + timedelta(seconds=frame_sec)
        frame_time_iso = frame_time.isoformat()
        
        # Supabase 신규 스키마 1:1 매핑 레코드
        count = len(pixels_map)
        aggregated_records.append({
            "clip_id": env_clip_id,                                   # vdoclip01m 참조 클립 ID (FK)
            "zone_id": env_zone_id,                                   # db_connector의 vdoclip01m 적재 시 필요
            "s3_clip_url": env_s3_url,                               # db_connector의 vdoclip01m 적재 시 필요
            "frame_id": int(frame_id),                               # 영상 프레임 번호
            "video_id": 1,                                           # 원본 영상 번호
            "total_count": count,                                    # 해당 프레임의 ROI 관제 구역 인원수
            "pixels_json": json.dumps(pixels_map, ensure_ascii=False), # 2D 픽셀 좌표 JSONb
            "bev_xyz_json": json.dumps(bev_xyz_map, ensure_ascii=False), # 3D 물리 좌표 JSONb
            "captured_at": frame_time_iso                            # 실제 캡처 시각
        })

    for gid, p_group in df_tracked.groupby('global_id'):
        traj = []
        for _, row in p_group.iterrows():
            traj.append({
                "frame": int(row['frame_id']),
                "x": round(float(row['bev_x_m']), 3),
                "y": round(float(row['bev_y_m']), 3),
                "z": round(float(row['bev_z_m']), 3)
            })
        trajectories_by_id[f"person_{int(gid)}"] = traj
# 5. 결과 파일 저장
pixels_json_path = os.path.join(RESULTS_DIR, "pedestrian_pixels_by_frame.json")
with open(pixels_json_path, "w", encoding="utf-8") as f:
    json.dump(pixels_by_frame_dict, f, ensure_ascii=False, indent=2)

bev_xyz_json_path = os.path.join(RESULTS_DIR, "pedestrian_bev_xyz_by_frame.json")
with open(bev_xyz_json_path, "w", encoding="utf-8") as f:
    json.dump(bev_xyz_by_frame_dict, f, ensure_ascii=False, indent=2)

pedaggr_json_path = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
with open(pedaggr_json_path, "w", encoding="utf-8") as f:
    json.dump(aggregated_records, f, ensure_ascii=False, indent=2)

csv_path = os.path.join(RESULTS_DIR, "pedaggr01h_aggregated.csv")
try:
    pd.DataFrame(aggregated_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
except PermissionError:
    csv_path = os.path.join(RESULTS_DIR, "pedaggr01h_aggregated_latest.csv")
    pd.DataFrame(aggregated_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[WARNING] pedaggr01h_aggregated.csv 파일이 엑셀 등에서 열려있어 pedaggr01h_aggregated_latest.csv 경로로 대체 저장되었습니다.")

sim_traj_path = os.path.join(RESULTS_DIR, "pedestrian_trajectories_by_id.json")
with open(sim_traj_path, "w", encoding="utf-8") as f:
    json.dump(trajectories_by_id, f, ensure_ascii=False, indent=2)

# 6. 기상청 API 연동 & Supabase DB (extfctr01h, vdoclip01m, pedaggr01h, mrkrisk01m) 일괄 적재
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "sensor_fusion_archive"))

weather_info = None
try:
    from weather_api import get_mangwon_weather
    fallback = os.environ.get("WEATHER_MODE", "SUNNY")
    weather_info = get_mangwon_weather(fallback_mode=fallback)
    print(f"\n[INFO] 기상청 정보 수집 완료: {weather_info['weather_condition']} ({weather_info['weather_label']}) - 가중치: {weather_info['weather_weight']}x")
except Exception as e:
    print(f"[WARNING] 기상청 API 연동 실패 (기본값 사용): {e}")

skip_db = os.environ.get("SKIP_DB_INSERT", "FALSE").upper() == "TRUE"
if skip_db:
    print(f"[INFO] SKIP_DB_INSERT=TRUE 설정에 따라 Supabase DB 직접 적재를 건너뜁니다.")
else:
    try:
        # pyrefly: ignore [missing-import]
        from utils.db_connector import bulk_insert_pedestrian_coordinate_json
        print(f"[INFO] Supabase DB (pedaggr01h / mrkrisk01m / extfctr01h) 데이터 적재 시도 중...")
        db_success = bulk_insert_pedestrian_coordinate_json(aggregated_records, weather_info=weather_info)
    except Exception as e:
        print(f"[WARNING] Supabase DB 적재 실패 또는 연동 미완료: {e}")

print(f"\n============================================================")
print(f"[SUCCESS] 최신 Supabase 스키마 & 기상 가중치 반영 완료!")
print(f" 1. 픽셀 좌표 JSON: {pixels_json_path}")
print(f" 2. BEV 3D 좌표 JSON: {bev_xyz_json_path}")
print(f" 3. pedaggr01h 최신 통합 JSON: {pedaggr_json_path}")
print(f" 4. pedaggr01h 최신 통합 CSV: {csv_path}")
print(f" 5. 시뮬팀 전용 ID 궤적 JSON: {sim_traj_path}")
if weather_info:
    print(f" 6. 적용된 날씨 가중치: {weather_info['weather_weight']}x ({weather_info['weather_label']})")
print(f"============================================================")


