# fusion_validation.py: CCTV BEV 좌표와 (CCTV 근처에 가상으로 매칭된) LiDAR 좌표를 유클리드 거리 기준으로 매칭하여 
# 센서 퓨전 성능을 검증하는 초기 테스트용 스크립트입니다.
import os
import pandas as pd
import numpy as np

# =========================================================================
# 1. 경로 설정
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv")

if not os.path.exists(CSV_PATH):
    print(f"⚠️ 에러: {CSV_PATH} 파일이 없습니다! video_to_bev.py를 먼저 실행해 주세요.")
    exit()

# 2. CCTV BEV 좌표 CSV 불러오기
cctv_df = pd.read_csv(CSV_PATH)
print(f"📂 CCTV 좌표 CSV 로드 완료! (총 {len(cctv_df)}건의 감지 데이터)")

# =========================================================================
# 3. 자동 융합 및 검증 함수
# =========================================================================
def run_automatic_fusion(threshold_m=0.8):
    unique_frames = sorted(cctv_df['frame'].unique())
    print(f"🚀 총 {len(unique_frames)}개 프레임에 대한 자동 센서 융합 검증 시작...\n")
    
    summary_list = []
    
    for frame_num in unique_frames:
        frame_cctv_data = cctv_df[cctv_df['frame'] == frame_num]
        cctv_pts = frame_cctv_data[['bev_x_m', 'bev_y_m']].to_numpy()
        cctv_count = len(cctv_pts)
        
        # -----------------------------------------------------------------
        # [참고] 실제 라이다 데이터 .bin이 연동되면 해당 좌표가 들어가는 위치입니다.
        # 현 단계에서는 매칭 테스트를 위해 CCTV 좌표 기반의 교차 검증 로직을 실행합니다.
        # -----------------------------------------------------------------
        # 가상의 라이다 DBSCAN 좌표 (CCTV 좌표 근처 + 라이다 단독 감지 객체 1~2명 포함)
        lidar_fake_pts = cctv_pts + np.random.normal(0, 0.15, cctv_pts.shape) # 매칭되는 라이다 점
        if frame_num % 5 == 0:  # 5프레임마다 라이다만 사각지대에서 1명 더 감지했다고 가정
            lidar_fake_pts = np.vstack([lidar_fake_pts, np.array([[12.0, 15.0]])])
            
        matched_count = 0
        cctv_only_count = 0
        lidar_matched = [False] * len(lidar_fake_pts)
        
        for c_pt in cctv_pts:
            matched = False
            for i, l_pt in enumerate(lidar_fake_pts):
                dist = np.linalg.norm(c_pt - l_pt)
                if dist <= threshold_m:
                    matched = True
                    lidar_matched[i] = True
                    break
            if matched:
                matched_count += 1
            else:
                cctv_only_count += 1
                
        lidar_only_count = lidar_matched.count(False)
        total_fusion_count = matched_count + cctv_only_count + lidar_only_count
        
        summary_list.append({
            'frame': frame_num,
            'cctv_count': cctv_count,
            'lidar_only_count': lidar_only_count,
            'matched_count': matched_count,
            'total_fusion_count': total_fusion_count
        })
        
        if frame_num % 50 == 0 or frame_num == unique_frames[-1]:
            print(f"📌 [Frame {frame_num:3d}] CCTV: {cctv_count}명 | 라이다 단독: {lidar_only_count}명 | 교차일치: {matched_count}명 ➔ 🎯 최종통합: {total_fusion_count}명")

    # 결과를 DataFrame으로 변환 및 CSV 저장
    result_df = pd.DataFrame(summary_list)
    save_result_path = os.path.join(RESULTS_DIR, "sensor_fusion_summary_second.csv")
    result_df.to_csv(save_result_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 모든 프레임 자동 융합 완료!")
    print(f"📊 최종 결과 요약 파일 저장됨: {save_result_path}")

# =========================================================================
# 4. 자동 실행 구문
# =========================================================================
if __name__ == "__main__":
    run_automatic_fusion()