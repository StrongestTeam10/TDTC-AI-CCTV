# test_CSR.py: CSRNet 모델로 비디오의 인원수를 예측하고, 정답 JSON 라벨 파일의 데이터와 비교하여 MSE, MAE 평가지표를 연산하고
# CSV로 저장하는 검증 스크립트입니다.
import os
import sys
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import cv2
from torchvision import models, transforms
from tqdm import tqdm

# =========================================================================
# 1. 경로 및 설정 (환경변수 주입 가능)
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("CSRNET_MODEL_PATH", os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth"))
VIDEO_PATH = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_cafe_output.mp4")
LABEL_DIR = os.environ.get("LABEL_DIR", r"E:\test\cctv_cafe_label")
OUTPUT_CSV = os.environ.get("CSR_ANSWER_CSV", os.path.join(RESULTS_DIR, "cctv_cafe_answer.csv"))

# 디바이스(GPU) 강제 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. CSRNet 모델 클래스 정의
# =========================================================================
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=None)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[0:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.backend(self.frontend(x))

# =========================================================================
# 3. GT 라벨 로딩 헬퍼 함수 (JSON 시나리오 파싱)
# =========================================================================
def load_scenario_label(label_dir):
    json_files = sorted(glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

    if not json_files:
        print(f"[WARNING] '{label_dir}' 경로에서 .json 라벨 파일을 찾을 수 없습니다!")
        return {}

    frame_gt_map = {}

    for json_path in json_files:
        data = None
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except Exception:
            try:
                with open(json_path, 'r', encoding='cp949') as f:
                    data = json.load(f)
            except Exception:
                continue

        if not data:
            continue

        # AI Hub 시나리오 통합 JSON 파싱 ("image" 리스트 구조)
        if isinstance(data, dict) and "image" in data and isinstance(data["image"], list):
            for idx, img_info in enumerate(data["image"], start=1):
                crowd_info = img_info.get("crowdinfo", {})
                if "counting" in crowd_info:
                    gt_cnt = int(crowd_info["counting"])
                elif "objects" in crowd_info:
                    gt_cnt = len(crowd_info["objects"])
                else:
                    gt_cnt = 0
                frame_gt_map[idx] = gt_cnt
        
        # 단일 프레임별 JSON 구조
        elif isinstance(data, dict):
            base_name = os.path.basename(json_path)
            try:
                # Indoor_EXCO001_002.json -> '002' -> 2
                frame_str = base_name.split('_')[-1].split('.')[0]
                idx = int(frame_str)
            except Exception:
                idx = len(frame_gt_map) + 1
                
            gt_cnt = 0
            if 'image' in data and isinstance(data['image'], dict):
                crowd_info = data['image'].get('crowdinfo', {})
                if 'counting' in crowd_info:
                    gt_cnt = int(crowd_info['counting'])
                elif 'objects' in crowd_info:
                    gt_cnt = len(crowd_info['objects'])
            
            if gt_cnt == 0:
                if 'counting' in data: gt_cnt = int(data['counting'])
                elif 'annotations' in data: gt_cnt = len(data['annotations'])
                elif 'objects' in data: gt_cnt = len(data['objects'])
                elif 'labels' in data: gt_cnt = len(data['labels'])
                
            frame_gt_map[idx] = gt_cnt

    return frame_gt_map

# =========================================================================
# 4. 메인 실행 루프
# =========================================================================
if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    print("="*80)
    print("      [CSRNet CCTV Count & Precision Evaluation Tool (test_CSR.py)]      ")
    print("="*80)
    print(f"- Device: {device}")
    print(f"- Model Weights Path: {MODEL_PATH}")
    print(f"- Input Video Path: {VIDEO_PATH}")
    print(f"- Label Folder:     {LABEL_DIR}")
    print("="*80)

    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 가중치 파일이 존재하지 않습니다: {MODEL_PATH}")
        sys.exit(1)

    model = CSRNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("[SUCCESS] CSRNet 가중치 파일 로드 성공!")

    # 2. GT 라벨 로드
    gt_map = load_scenario_label(LABEL_DIR)
    if gt_map:
        print(f"[SUCCESS] 총 {len(gt_map)}개 프레임의 정답(GT) 정보 로드 성공!")
    else:
        print("[WARNING] 정답 라벨을 불러오지 못했습니다. 정확도/정밀도 계산은 스킵됩니다.")

    # 3. 비디오 로드
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 비디오 파일이 존재하지 않습니다: {VIDEO_PATH}")
        sys.exit(1)

    # 한글 경로 지원 임시 비디오 파일 리더 처리
    is_unicode = False
    try:
        VIDEO_PATH.encode('ascii')
    except UnicodeEncodeError:
        is_unicode = True
        
    temp_read_path = None
    if is_unicode:
        import shutil
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_csr_{os.getpid()}.mp4")
        shutil.copy(VIDEO_PATH, temp_path)
        cap = cv2.VideoCapture(temp_path)
        temp_read_path = temp_path
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] 비디오 파일을 열 수 없습니다.")
        if temp_read_path and os.path.exists(temp_read_path):
            os.remove(temp_read_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Analyzing Video (Total {total_frames} frames)...")

    # 전처리 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results_data = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Processing Pipeline Inference", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 720p 해상도 고정하여 스케일 일치화
        frame_720p = cv2.resize(frame, (1280, 720))
        img_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.clamp(output, min=0)
            pred_count = output.sum().item()

        # 정수값 인원 산출
        pred_int = int(np.round(pred_count))
        gt_int = gt_map.get(frame_idx, None)

        record = {
            "frame": frame_idx,
            "pred_count": pred_int,
            "pred_raw": round(pred_count, 2)
        }

        if gt_int is not None:
            record["gt_count"] = gt_int
            error = abs(gt_int - pred_int)
            record["error"] = error
            
            # 수치 예측 평가를 위한 가상 분류 기준 정의 (오차범위 실제 인원의 15% 이내 또는 +-3명 이내)
            tolerance = max(3, int(gt_int * 0.15))
            if error <= tolerance:
                record["classification"] = "TP"  # 허용 오차 내 탐지 (True Positive)
            elif pred_int < gt_int:
                record["classification"] = "FN"  # 인원 과소 탐지 (False Negative)
            else:
                record["classification"] = "FP"  # 인원 과다 탐지 (False Positive)
        else:
            record["gt_count"] = "N/A"
            record["error"] = "N/A"
            record["classification"] = "N/A"

        results_data.append(record)
        pbar.update(1)

    cap.release()
    pbar.close()

    if temp_read_path and os.path.exists(temp_read_path):
        try:
            os.remove(temp_read_path)
        except Exception:
            pass

    # 5. CSV 저장
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nSaved evaluation CSV to: {OUTPUT_CSV}")

    # 6. 통계 지표 산출 및 출력
    if gt_map:
        df_eval = df[df["gt_count"] != "N/A"].copy()
        df_eval["gt_count"] = df_eval["gt_count"].astype(int)
        df_eval["pred_count"] = df_eval["pred_count"].astype(int)
        df_eval["error"] = df_eval["error"].astype(int)

        gt_arr = df_eval["gt_count"].values
        pred_arr = df_eval["pred_count"].values
        errors = df_eval["error"].values

        # 수치 오차
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 탐지 정확도
        sum_gt = np.sum(gt_arr)
        sum_pred = np.sum(pred_arr)
        accuracy = (sum_pred / sum_gt * 100) if sum_gt > 0 else 0

        # 분류 기반 Precision / Recall / F1-Score
        tp_count = len(df_eval[df_eval["classification"] == "TP"])
        fp_count = len(df_eval[df_eval["classification"] == "FP"])
        fn_count = len(df_eval[df_eval["classification"] == "FN"])

        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0
        recall = (tp_count / (tp_count + fn_count) * 100) if (tp_count + fn_count) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        print("\n" + "="*80)
        print(" [CSRNet Prediction Evaluation Report] ")
        print("="*80)
        print(f"- Total Analyzed Frames: {len(df_eval)} frames")
        print(f"- Actual Total Count (GT): {sum_gt:,} people (Avg: {np.mean(gt_arr):.2f})")
        print(f"- Pred Total Count:       {sum_pred:,} people (Avg: {np.mean(pred_arr):.2f})")
        print("-"*80)
        print(f"- Mean Absolute Error (MAE):  {mae:.2f} people")
        print(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"- Total Detection Ratio (Acc):   {accuracy:.2f}%")
        print("-"*80)
        print(" [Precision / Recall / F1-Score classification metrics] ")
        print("   * Threshold: Within +/-15% of GT or +/-3 people")
        print(f"   * True Positive (TP - inside tolerance): {tp_count}")
        print(f"   * False Positive (FP - over-counting):  {fp_count}")
        print(f"   * False Negative (FN - under-counting): {fn_count}")
        print(f"   - Precision: {precision:.2f}%")
        print(f"   - Recall:    {recall:.2f}%")
        print(f"   - F1-Score:  {f1_score:.2f}%")
        print("="*80)
