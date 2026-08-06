# evaluate_with_labels.py: 시나리오 통합 JSON 또는 개별 JSON에서 프레임별 정답 인원을 추출하여 CCTV 및 센서 퓨전 결과와 비교하고, 
# 종합 정확도 지표(RMSE, MAE 등)를 계산하는 평가 스크립트입니다.
import os
import glob
import json
import numpy as np
import pandas as pd

# =========================================================================
# 1. 경로 설정
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 검증할 CSV 결과 파일 (CSRNet BEV 좌표 CSV 또는 센서 퓨전 CSV)
CCTV_CSV_PATH = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv"))
FUSION_CSV_PATH = os.environ.get("FUSION_SUMMARY_CSV", os.path.join(RESULTS_DIR, "sensor_fusion_summary_v2.csv"))

# 🔥 통합 JSON 파일이 위치한 폴더
LABEL_DIR = os.environ.get("LABEL_DIR", r"E:\test\cctv_cafe_label")

# =========================================================================
# 2. 시나리오 통합 JSON 파싱 함수
# =========================================================================
def load_scenario_label(label_dir):
    """
    단일/다중 JSON 파일에서 프레임별 GT(정답) 카운트를 추출
    """
    # 1. recursive 하위 탐색으로 json 파일 검색
    json_files = sorted(glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

    if not json_files:
        print(f"❌ '{label_dir}' 경로에서 .json 라벨 파일을 찾을 수 없습니다!")
        return []

    frame_gt_list = []

    for json_path in json_files:
        # 인코딩 예외 처리 (utf-8-sig -> cp949)
        data = None
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except Exception:
            try:
                with open(json_path, 'r', encoding='cp949') as f:
                    data = json.load(f)
            except Exception as e:
                continue

        if not data:
            continue

        # 💡 AI Hub 시나리오 구조 (단일 JSON 파일 내 "image" 리스트 구조)
        if isinstance(data, dict) and "image" in data and isinstance(data["image"], list):
            print(f"📜 시나리오 통합 JSON 로드 완료: {os.path.basename(json_path)}")
            for idx, img_info in enumerate(data["image"], start=1):
                img_name = img_info.get("imagename", f"frame_{idx}")
                crowd_info = img_info.get("crowdinfo", {})
                
                # counting 항목 추출, 없으면 objects 개수로 대체
                if "counting" in crowd_info:
                    gt_cnt = int(crowd_info["counting"])
                elif "objects" in crowd_info:
                    gt_cnt = len(crowd_info["objects"])
                else:
                    gt_cnt = 0

                frame_gt_list.append({
                    'frame': idx,
                    'image_name': img_name,
                    'gt_count': gt_cnt
                })
        
        # 일반 단일 프레임별 JSON 구조인 경우
        elif isinstance(data, dict):
            idx = len(frame_gt_list) + 1
            gt_cnt = 0
            if 'counting' in data: gt_cnt = data['counting']
            elif 'annotations' in data: gt_cnt = len(data['annotations'])
            elif 'objects' in data: gt_cnt = len(data['objects'])
            elif 'labels' in data: gt_cnt = len(data['labels'])

            frame_gt_list.append({
                'frame': idx,
                'image_name': os.path.basename(json_path),
                'gt_count': gt_cnt
            })

    return frame_gt_list

# =========================================================================
# 3. 라벨 대비 예측 데이터 비교 연산
# =========================================================================
print("🔍 [시나리오 데이터셋 기반 AI 정밀 평가 가동]")
print(f"📂 대상 경로: {LABEL_DIR}\n")

gt_records = load_scenario_label(LABEL_DIR)

if not gt_records:
    print("❌ 정답 데이터 로드 실패. 종료합니다.")
    exit()

print(f"✅ 총 {len(gt_records)}개 프레임의 정답(GT) 라벨 파싱 성공!\n")

# CSV 예측 데이터 로드
cctv_df = pd.read_csv(CCTV_CSV_PATH) if os.path.exists(CCTV_CSV_PATH) else None
fusion_df = pd.read_csv(FUSION_CSV_PATH) if os.path.exists(FUSION_CSV_PATH) else None

eval_summary = []

for record in gt_records:
    f_idx = record['frame']
    gt_cnt = record['gt_count']

    # 1) CCTV BEV CSV 탐지 인원
    cctv_cnt = 0
    if cctv_df is not None and 'frame' in cctv_df.columns:
        cctv_cnt = len(cctv_df[cctv_df['frame'] == f_idx])

    # 2) 센서 퓨전 CSV 탐지 인원
    fusion_cnt = cctv_cnt
    if fusion_df is not None and 'frame' in fusion_df.columns:
        f_match = fusion_df[fusion_df['frame'] == f_idx]
        if not f_match.empty and 'total_fusion_count' in f_match.columns:
            fusion_cnt = int(f_match['total_fusion_count'].values[0])

    eval_summary.append({
        'frame': f_idx,
        'image_name': record['image_name'],
        'GT_Count': gt_cnt,
        'CCTV_Pred': cctv_cnt,
        'Fusion_Pred': fusion_cnt,
        'Error_CCTV': abs(gt_cnt - cctv_cnt),
        'Error_Fusion': abs(gt_cnt - fusion_cnt)
    })

# =========================================================================
# 4. 오차 및 정확도 통계 산출
# =========================================================================
df_eval = pd.DataFrame(eval_summary)

gt_arr = df_eval['GT_Count'].values
cctv_arr = df_eval['CCTV_Pred'].values
fusion_arr = df_eval['Fusion_Pred'].values

mae_cctv = np.mean(df_eval['Error_CCTV'])
rmse_cctv = np.sqrt(np.mean((gt_arr - cctv_arr) ** 2))
acc_cctv = (np.sum(cctv_arr) / np.sum(gt_arr) * 100) if np.sum(gt_arr) > 0 else 0

mae_fusion = np.mean(df_eval['Error_Fusion'])
rmse_fusion = np.sqrt(np.mean((gt_arr - fusion_arr) ** 2))
acc_fusion = (np.sum(fusion_arr) / np.sum(gt_arr) * 100) if np.sum(gt_arr) > 0 else 0

# =========================================================================
# 5. 저장 및 보고서 출력
# =========================================================================
EVAL_SAVE_PATH = os.path.join(RESULTS_DIR, "label_evaluation_summary.csv")
df_eval.to_csv(EVAL_SAVE_PATH, index=False, encoding='utf-8-sig')

print("=" * 65)
print("📊 [북촌카페 시나리오 정답 라벨 대비 종합 평가 보고서]")
print("=" * 65)
print(f"📄 총 검증 프레임 수: {len(df_eval)}개")
print(f"👥 전체 정답(GT) 총 인원: {np.sum(gt_arr)}명")
print("-" * 65)
print("1️⃣ CCTV 단독 (CSRNet BEV) 평가:")
print(f"   - 평균 절대 오차 (MAE): ±{mae_cctv:.2f}명")
print(f"   - 제곱근 평균 오차 (RMSE): {rmse_cctv:.2f}")
print(f"   - 정답 대비 인원 탐지율: {acc_cctv:.2f}%")
print("-" * 65)
print("2️⃣ CCTV + LiDAR 하이브리드 센서 퓨전 평가:")
print(f"   - 평균 절대 오차 (MAE): ±{mae_fusion:.2f}명")
print(f"   - 제곱근 평균 오차 (RMSE): {rmse_fusion:.2f}")
print(f"   - 정답 대비 인원 탐지율: {acc_fusion:.2f}%")
print("=" * 65)
print(f"📁 프레임별 상세 검증 표 저장 완료: {EVAL_SAVE_PATH}")
print("=" * 65)