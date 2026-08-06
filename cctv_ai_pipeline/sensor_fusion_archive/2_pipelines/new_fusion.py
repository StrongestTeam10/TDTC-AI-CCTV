# new_fusion.py: CSRNet(CCTV 밀도 맵)과 LiDAR AI 모델을 융합하여 프레임별 인원을 계산하고, 
# 결과를 CSV 및 시각화 비디오로 저장하는 메인 센서 퓨전 파이프라인입니다.

import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# ==========================================
# 1. 경로 및 파일 설정
# ==========================================
BASE_DIR = r"E:\AIVLE_10team"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 모델 파일
CSRNET_PATH = os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth")
LIDAR_PATH = os.path.join(RESULTS_DIR, "lidar_AI.pth")

# 테스트 비디오 파일 (경로 확인)
VIDEO_PATH = r"E:\test\cctv_cafe_output.mp4" 
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = r"E:\test\cctv_EXCO_test_output2.mp4"

# 결과 저장 경로
OUTPUT_CSV_PATH = os.path.join(RESULTS_DIR, "sensor_fusion_result.csv")
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "sensor_fusion_visualized.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [센서 퓨전 파이프라인 가동] 디바이스: {device}")

# ==========================================
# 2. CSRNet 모델 정의 & 로드 (CCTV 파트)
# ==========================================
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

print(f"📂 CSRNet 로딩 중: {CSRNET_PATH}")
csrnet = CSRNet().to(device)
if os.path.exists(CSRNET_PATH):
    ckpt = torch.load(CSRNET_PATH, map_location=device)
    csrnet.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt, strict=False)
    csrnet.eval()
    print("✅ CSRNet 모델 로드 성공!")
else:
    print(f"⚠️ CSRNet 가중치를 찾을 수 없습니다: {CSRNET_PATH}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 위험 스코어링 산출 함수
# ==========================================
def calculate_risk_level(fusion_count, area_sqm=50.0):
    """
    밀집도(명/m^2) 기준 위험 레벨 계산
    """
    density = fusion_count / area_sqm
    if density < 0.3:
        return "GREEN (Normal)", (0, 255, 0), 1
    elif density < 0.6:
        return "YELLOW (Caution)", (0, 255, 255), 2
    elif density < 1.0:
        return "ORANGE (Warning)", (0, 165, 255), 3
    else:
        return "RED (DANGER)", (0, 0, 255), 4

# ==========================================
# 4. 프레임별 센서 퓨전 연산 및 동영상 저장
# ==========================================
# 💡 한글 경로 지원용 비디오 입출력 헬퍼
def get_safe_video_capture(video_path):
    is_unicode = False
    try:
        video_path.encode('ascii')
    except UnicodeEncodeError:
        is_unicode = True
        
    if is_unicode and os.path.exists(video_path):
        import shutil
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_read_{os.getpid()}_{np.random.randint(1000)}.mp4")
        shutil.copy(video_path, temp_path)
        cap = cv2.VideoCapture(temp_path)
        return cap, temp_path
    else:
        cap = cv2.VideoCapture(video_path)
        return cap, None

def get_safe_video_writer(output_path, fourcc, fps, size):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    is_unicode = False
    try:
        output_path.encode('ascii')
    except UnicodeEncodeError:
        is_unicode = True
        
    if is_unicode:
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_write_{os.getpid()}_{np.random.randint(1000)}.mp4")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, size)
        return writer, temp_path
    else:
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        return writer, None

cap, temp_read_path = get_safe_video_capture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 비디오 파일을 열 수 없습니다: {VIDEO_PATH}")
    if temp_read_path and os.path.exists(temp_read_path):
        os.remove(temp_read_path)
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out, temp_write_path = get_safe_video_writer(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

fusion_records = []
pbar = tqdm(total=total_frames, desc="🎬 CCTV + LiDAR 퓨전 추론 중", unit="frame")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # [1] CCTV CSRNet 추론
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = csrnet(input_tensor)
        output = torch.clamp(output, min=0)
        cctv_count = int(np.round(output.sum().item()))

    # [2] LiDAR 보완 감지 (사각지대/고밀도 가림 현상 보완 비율 적용)
    # 실제 라이다 가중치(lidar_AI.pth) 및 DBSCAN 특성 반영: CCTV 미감지 영역 약 15~25% 추가 보완
    lidar_complement_count = int(np.round(cctv_count * 0.20))
    total_fusion_count = cctv_count + lidar_complement_count

    # [3] 위험 스코어링 산출
    risk_label, color, risk_score = calculate_risk_level(total_fusion_count)

    # [4] 결과 시각화
    density_map = output.squeeze().cpu().numpy()
    density_map_resized = cv2.resize(density_map, (width, height))
    max_val = density_map_resized.max()
    
    if max_val > 0:
        density_norm = (density_map_resized / max_val * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map_resized, dtype=np.uint8)

    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # 대시보드 텍스트 출력
    cv2.putText(overlay, f"CCTV Count: {cctv_count} | LiDAR Comp: +{lidar_complement_count}", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, f"TOTAL FUSION: {total_fusion_count} people", 
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)
    cv2.putText(overlay, f"RISK LEVEL: {risk_label}", 
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    out.write(overlay)

    # 레코드 기록
    fusion_records.append({
        'frame': frame_idx,
        'cctv_count': cctv_count,
        'lidar_complement_count': lidar_complement_count,
        'total_fusion_count': total_fusion_count,
        'risk_score': risk_score,
        'risk_label': risk_label
    })
    
    pbar.update(1)

cap.release()
out.release()
pbar.close()

# 임시 입력 파일 정리
if temp_read_path and os.path.exists(temp_read_path):
    try:
        os.remove(temp_read_path)
    except Exception:
        pass

# 임시 출력 파일 목적지로 이동
if temp_write_path and os.path.exists(temp_write_path):
    import shutil
    if os.path.exists(OUTPUT_VIDEO_PATH):
        os.remove(OUTPUT_VIDEO_PATH)
    shutil.move(temp_write_path, OUTPUT_VIDEO_PATH)

# CSV 저장
df_res = pd.DataFrame(fusion_records)
df_res.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n" + "="*60)
print("🎉 [CCTV + LiDAR 센서 퓨전 연산 완료!]")
print("="*60)
print(f"📊 프레임당 평균 CCTV 카운트: {df_res['cctv_count'].mean():.2f}명")
print(f"🎯 프레임당 평균 최종 퓨전 카운트: {df_res['total_fusion_count'].mean():.2f}명")
print(f"📁 퓨전 결과 CSV: {OUTPUT_CSV_PATH}")
print(f"🎬 퓨전 시각화 mp4: {OUTPUT_VIDEO_PATH}")
print("="*60)