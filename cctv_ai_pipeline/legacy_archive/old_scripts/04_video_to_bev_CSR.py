# video_to_bev_CSR.py: CCTV 비디오에서 CSRNet 모델을 통해 사람의 2D 위치를 추출한 후, 
# 호모그래피 변환을 이용해 LiDAR BEV 2D 물리 좌표로 매핑하여 CSV로 저장하는 스크립트입니다.

import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import scipy.ndimage as ndimage
from tqdm import tqdm

# =========================================================================
# 1. 경로 및 설정 (CSRNet 가중치 & 테스트 비디오)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("CSRNET_MODEL_PATH", os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth"))

# 테스트 비디오 경로 설정
VIDEO_PATH = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_cafe_output.mp4")
if not os.path.exists(VIDEO_PATH) and "OUTPUT_MP4" not in os.environ:
    VIDEO_PATH = r"E:\test\cctv_EXCO_test_output2.mp4"

CSV_SAVE_PATH = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [CSRNet -> BEV 좌표 추출기 가동] 사용 디바이스: {device}")

# =========================================================================
# 2. 석훈님 CSRNet 모델 클래스 정의
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

print(f"📂 CSRNet 모델 로딩 중: {MODEL_PATH}")
model = CSRNet().to(device)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt, strict=False)
    model.eval()
    print("✅ CSRNet 가중치 로드 완료!")
else:
    print(f"❌ 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")
    exit()

# =========================================================================
# 3. 호모그래피 변환 행렬 & Peak 추출 함수
# =========================================================================
DATASET_TYPE = os.environ.get("DATASET_TYPE", "EXCO")

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

def extract_peaks_from_density(density_map, threshold=0.01):
    """CSRNet 밀도 히트맵에서 보행자 중심점 Peak (u, v) 좌표 추출 (하위 호환용)"""
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(density_map, neighborhood_size)
    maxima = (density_map == data_max)
    data_min = ndimage.minimum_filter(density_map, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[~diff] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    peaks = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2.0
        y_center = (dy.start + dy.stop - 1) / 2.0
        peaks.append((x_center, y_center))
    return peaks

def extract_peaks_from_density_gpu(density_tensor, threshold=0.01):
    """GPU 텐서 상에서 5x5 로컬 맥시마(Peak)를 고속 연산하여 좌표 추출"""
    # 5x5 Max/Min 필터링을 GPU 상에서 고속 연산 (min_pool은 음수 트릭 적용)
    data_max = F.max_pool2d(density_tensor, kernel_size=5, stride=1, padding=2)
    data_min = -F.max_pool2d(-density_tensor, kernel_size=5, stride=1, padding=2)
    
    maxima_mask = (density_tensor == data_max)
    diff_mask = (data_max - data_min) > threshold
    
    # 최종 로컬 맥시마 텐서 마스크 (bool)
    keep = maxima_mask & diff_mask
    
    # 크기가 매우 작은 저해상도 (예: 160x90) 마스크만 CPU numpy로 전환
    maxima_cpu = keep.squeeze().cpu().numpy()
    
    # 저해상도 스케일 상에서 매우 가볍게 라벨링 수행 (연산 속도 64배 이상 단축)
    labeled, num_objects = ndimage.label(maxima_cpu)
    slices = ndimage.find_objects(labeled)
    
    peaks = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2.0
        y_center = (dy.start + dy.stop - 1) / 2.0
        peaks.append((x_center, y_center))
    return peaks

# =========================================================================
# 4. 비디오 추론 및 BEV 좌표 추출
# =========================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 💡 한글 경로 지원용 비디오 로드 헬퍼
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

# 분석 타겟 FPS 설정 (환경변수 TARGET_FPS 우선 적용, 기본값 10.0)
target_fps = float(os.environ.get("TARGET_FPS", "10.0"))
skip_interval = max(1, int(round(fps / target_fps))) if fps > target_fps else 1

data_list = []
frame_count = 0

print(f"🎬 [CSRNet] 비디오 프레임별 BEV 좌표 연산 시작: {VIDEO_PATH}")
print(f"📊 원본 FPS: {fps:.2f} -> 분석 타겟 FPS: {target_fps} (샘플링 간격: {skip_interval}프레임당 1프레임)\n")
pbar = tqdm(total=total_frames, desc="🎬 BEV 좌표 변환 중", unit="frame")

while cap.isOpened():
    frame_count += 1
    
    # 10/15 FPS 샘플링을 위해 프레임 스키핑 (cap.grab()을 통한 초고속 포인터 이동)
    if (frame_count - 1) % skip_interval != 0:
        if not cap.grab():
            break
        pbar.update(1)
        continue
        
    ret, frame = cap.read()
    # 720p 해상도로 리사이즈 (save_CSR_result.py 와 추론 해상도 일치화)
    frame_720p = cv2.resize(frame, (1280, 720))
    img_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, min=0)
        
        # GPU에서 직접 5x5 로컬 맥시마 Peak 추출
        threshold_val = 0.0015 if DATASET_TYPE == "MALL" else 0.005
        raw_peaks = extract_peaks_from_density_gpu(output, threshold=threshold_val)

    # 저해상도 좌표를 추론 해상도(720p)에 맞추어 스케일 복원
    h_out, w_out = output.shape[2], output.shape[3]
    scale_x = 1280.0 / w_out
    scale_y = 720.0 / h_out
    
    peaks = []
    for px_raw, py_raw in raw_peaks:
        u = px_raw * scale_x
        v = py_raw * scale_y
        peaks.append((u, v))

    person_id = 0
    for u, v in peaks:
        person_id += 1
        # BEV 3D 공간 미터 좌표 변환
        bev_x, bev_y = transform_pixel_to_bev(u, v, H_MATRIX)
        
        data_list.append({
            'frame': frame_count,
            'timestamp_sec': round(frame_count / fps, 2),
            'person_id': person_id,
            'pixel_u': round(u, 2),
            'pixel_v': round(v, 2),
            'bev_x_m': round(bev_x, 2),
            'bev_y_m': round(bev_y, 2)
        })

    pbar.update(1)

cap.release()
pbar.close()

if temp_read_path and os.path.exists(temp_read_path):
    try:
        os.remove(temp_read_path)
    except Exception:
        pass

# CSV 저장
df = pd.DataFrame(data_list)
df.to_csv(CSV_SAVE_PATH, index=False)

print("\n" + "="*50)
print("🎉 [CSRNet 기반] CCTV BEV 좌표 CSV 변환 완벽 성공!")
print(f"📁 저장 경로: {CSV_SAVE_PATH}")
print(f"👥 총 추출된 보행자 좌표 수: {len(df)}개")
print("="*50)