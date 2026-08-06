# analyze_mangwon_labels.py
# ===================================================================================================
# [목적 및 역할]
# 본 스크립트는 망원시장 CCTV 이미지에 매핑된 JSON 라벨 데이터를 데이터프레임으로 변환 및 전처리하고,
# 픽셀 좌표를 BEV 물리 좌표로 투영하여 분석을 수행하는 유틸리티 스크립트입니다.
# ===================================================================================================

import os
import glob
import json
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

# =========================================================================
# 1. 설정 및 경로 정의
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
LABEL_DIR_PATTERN = r"E:\test\망원시장_라벨_*"  # 분석 대상 JSON 라벨 폴더 패턴
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. 호모그래피 변환 행렬 (EXCO/Mall 기준 임시값 ➔ 추후 망원시장 캘리브레이션에 맞춰 수정 필요)
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

# =========================================================================
# 3. JSON 라벨 파싱 및 전처리 함수 (다중 폴더 지원)
# =========================================================================
def parse_mangwon_labels(label_dir_pattern):
    # 패턴에 매칭되는 모든 라벨 폴더 목록 가져오기
    label_dirs = sorted(glob.glob(label_dir_pattern))
    
    if not label_dirs:
        print(f"[ERROR] '{label_dir_pattern}' 패턴에 매칭되는 폴더가 존재하지 않습니다.")
        return None, None
        
    summary_list = []
    pedestrian_list = []
    
    print(f"[INFO] 발견된 라벨 폴더 목록: {[os.path.basename(d) for d in label_dirs]}")
    
    for label_dir in label_dirs:
        dir_name = os.path.basename(label_dir)
        
        # 폴더명에서 시퀀스 ID 추출 (예: 망원시장_라벨_1 -> 1, 망원시장_라벨_10 -> 10)
        try:
            sequence_id = int(dir_name.split('_')[-1])
        except Exception:
            sequence_id = dir_name
            
        json_paths = sorted(glob.glob(os.path.join(label_dir, "*.json")))
        print(f"[INFO] 폴더 파싱 중: {dir_name} ({len(json_paths)}개 JSON 파일)")
        
        for path in json_paths:
            base_name = os.path.basename(path)
            
            # 파일명에서 프레임 번호 추출 (예: Outdoor_서울망원시장001_002.json -> 2)
            try:
                frame_str = base_name.split('_')[-1].split('.')[0]
                frame_id = int(frame_str)
            except Exception:
                frame_id = 0
                
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                
            # crowdinfo 헤더 파싱
            crowd_info = data.get("image", {}).get("crowdinfo", {})
            counting = crowd_info.get("counting", 0)
            collectiveness = crowd_info.get("collectiveness", 0)
            stability = crowd_info.get("stability", 0)
            uniformity = crowd_info.get("uniformity", 0)
            
            # 1) 프레임별 요약 테이블 구성
            summary_list.append({
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "filename": base_name.replace(".json", ".jpg"),
                "true_count": counting,
                "collectiveness": collectiveness,
                "stability": stability,
                "uniformity": uniformity
            })
            
            # 2) 보행자 개별 픽셀 좌표 테이블 구성
            objects = crowd_info.get("objects", [])
            for p_idx, obj in enumerate(objects, start=1):
                direction = obj.get("directionindex", [])
                if len(direction) >= 2:
                    pixel_x, pixel_y = direction[0], direction[1]
                    
                    # 호모그래피 변환을 통한 BEV 2D 미터 좌표 변환
                    bev_x, bev_y = transform_pixel_to_bev(pixel_x, pixel_y, H_MATRIX)
                    
                    pedestrian_list.append({
                        "sequence_id": sequence_id,
                        "frame_id": frame_id,
                        "person_id": p_idx,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "bev_x_m": bev_x,
                        "bev_y_m": bev_y
                    })
                
    df_summary = pd.DataFrame(summary_list)
    df_pedestrians = pd.DataFrame(pedestrian_list)
    
    return df_summary, df_pedestrians

# =========================================================================
# 4. 분석 수행 예제
# =========================================================================
if __name__ == "__main__":
    df_summary, df_pedestrians = parse_mangwon_labels(LABEL_DIR_PATTERN)
    
    if df_summary is not None and df_pedestrians is not None:
        print("\n" + "="*50)
        print("[SUCCESS] [망원시장 다중 시퀀스 라벨 데이터프레임 전처리 완료]")
        print("="*50)
        
        # 1) 기초 통계량 출력
        total_sequences = df_summary['sequence_id'].nunique()
        print(f"- 총 분석 시퀀스(폴더) 수: {total_sequences} 개")
        print(f"- 총 분석 프레임 수: {len(df_summary)} 개")
        print(f"- 총 감지된 보행자 포인트 수: {len(df_pedestrians)} 건")
        print(f"- 프레임당 평균 인원수: {df_summary['true_count'].mean():.2f} 명")
        print(f"- 프레임당 최대 인원수: {df_summary['true_count'].max()} 명")
        print(f"- 프레임당 최소 인원수: {df_summary['true_count'].min()} 명")
        
        # 시퀀스별 평균 인원 분석
        print("\n[INFO] 시퀀스(폴더)별 평균 실제 인원 통계:")
        seq_means = df_summary.groupby('sequence_id')['true_count'].mean()
        for seq_id, mean_val in seq_means.items():
            print(f" - 시퀀스 {seq_id} 평균 인원: {mean_val:.2f} 명")
        
        # 2) 특정 가상의 ROI(위험 관리 구역) 내 실제 밀도 분석 시뮬레이션
        # 가상의 사거리 구역 정의 (X: -5m ~ 5m, Y: 5m ~ 15m)
        roi_polygon = Polygon([(-5.0, 5.0), (5.0, 5.0), (5.0, 15.0), (-5.0, 15.0)])
        roi_area = roi_polygon.area  # 10m x 10m = 100㎡
        
        print("\n[INFO] [임시 ROI 밀집도 분석 시뮬레이션]")
        print(f" - 분석 대상 구역: 가상의 사거리 (면적: {roi_area:.1f}㎡)")
        
        # 각 보행자 좌표가 ROI에 포함되는지 체크
        df_pedestrians['in_roi'] = df_pedestrians.apply(
            lambda row: roi_polygon.contains(Point(row['bev_x_m'], row['bev_y_m'])), axis=1
        )
        
        # 프레임별 ROI 내 실제 인원 수 집계 (시퀀스별, 프레임별로 고유 분류)
        roi_counts = df_pedestrians[df_pedestrians['in_roi']].groupby(['sequence_id', 'frame_id']).size().reset_index(name='roi_count')
        
        # 프레임 요약 데이터에 다중키로 병합 (Merge)
        df_summary = pd.merge(df_summary, roi_counts, on=['sequence_id', 'frame_id'], how='left')
        df_summary['roi_count'] = df_summary['roi_count'].fillna(0).astype(int)
        df_summary['roi_density_per_m2'] = df_summary['roi_count'] / roi_area
        
        print(f" - ROI 내 전체 프레임당 평균 실제 인원: {df_summary['roi_count'].mean():.2f} 명")
        print(f" - ROI 내 전체 프레임당 최대 실제 인원: {df_summary['roi_count'].max()} 명")
        print(f" - ROI 내 최대 실제 밀도: {df_summary['roi_density_per_m2'].max():.3f} 명/㎡")
        
        # 3) 전처리 완료된 데이터 로컬 CSV 저장
        summary_csv = os.path.join(RESULTS_DIR, "mangwon_label_summary.csv")
        pedestrian_csv = os.path.join(RESULTS_DIR, "mangwon_label_pedestrians.csv")
        
        df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        df_pedestrians.to_csv(pedestrian_csv, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"[FILE] [결과 CSV 통합 저장 완료]")
        print(f" - 요약 데이터 저장 완료: {summary_csv}")
        print(f" - 보행자 좌표 저장 완료: {pedestrian_csv}")
        print("="*50)
    else:
        print("[ERROR] 데이터프레임 파싱을 실패하였습니다.")
