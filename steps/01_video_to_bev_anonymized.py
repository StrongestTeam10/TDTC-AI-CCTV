# video_to_bev_CSR.py: CCTV 비디오에서 CSRNet 모델을 통해 사람의 2D 위치를 추출한 후, 
# 호모그래피 변환을 이용해 LiDAR BEV 2D 물리 좌표로 매핑하여 CSV로 저장하는 스크립트입니다.

import os
import json
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

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# =========================================================================
# 1. 경로 및 설정 (CSRNet 가중치 & 테스트 비디오)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("CSRNET_MODEL_PATH", os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth"))
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth")

# YOLO 모델 경로 (yolo11n.pt 최우선)
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", os.path.join(MODELS_DIR, "yolo11n.pt"))
if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "bestYOLOm5080model.pt")
if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = "yolo11n.pt"

# 테스트 비디오 경로 설정
VIDEO_PATH = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_cafe_output.mp4")
if not os.path.exists(VIDEO_PATH) and "OUTPUT_MP4" not in os.environ:
    VIDEO_PATH = r"E:\test\cctv_EXCO_test_output2.mp4"

CSV_SAVE_PATH = os.environ.get("CCTV_BEV_CSV", os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv"))
CSR_RESULT_MP4 = os.environ.get("CSR_RESULT_MP4", os.path.join(RESULTS_DIR, "cctv_result_anonymized.mp4"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [CSRNet + YOLO11n -> BEV 및 모자이크 파이프라인 가동] 사용 디바이스: {device}")

# 가우시안 모자이크 비식별화 함수
def apply_mosaic(img, x1, y1, x2, y2):
    """지정된 바운딩 박스 영역에 부드러운 가우시안 블러 비식별화(모자이크) 적용"""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return img
        
    roi = img[y1:y2, x1:x2]
    kernel_w = max(3, (w // 2) * 2 + 1)
    kernel_h = max(3, (h // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (kernel_w, kernel_h), sigmaX=15, sigmaY=15)
    img[y1:y2, x1:x2] = blurred
    return img

yolo_model = None
if yolo_available and os.path.exists(YOLO_MODEL_PATH):
    try:
        print(f"📂 YOLO11n 비식별화 모델 로딩 중: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO11n 모델 로딩 완벽 성공!")
    except Exception as e:
        print(f"⚠️ YOLO11n 로드 경고: {e}")


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
zone_id = str(os.environ.get("ZONE_ID", "1"))

H_MATRIX = None
ROI_2D_POLY = None
try:
    # e:/AIVLE_10team/ai_pipeline/cctv_upload/core/zones_config.json 경로 탐색
    config_path = os.path.join(os.path.dirname(BASE_DIR), "core", "zones_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, "core", "zones_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            zones_config = json.load(f)
            zone_meta = zones_config.get(zone_id, zones_config.get("1", {}))
            H_MATRIX = np.array(zone_meta.get("h_matrix"))
            ROI_2D_POLY = np.array(zone_meta.get("roi_polygon"), dtype=np.int32)
            print(f"✅ [CONFIG] zones_config.json에서 Zone {zone_id} 캘리브레이션 행렬 및 ROI 로드 완료!")
except Exception as e:
    print(f"⚠️ [CONFIG] zones_config.json 로드 중 오류 발생, 기본값 사용: {e}")

if H_MATRIX is None:
    H_MATRIX = np.array([
        [ 0.015, -0.002, -8.50],
        [ 0.001,  0.022, -2.10],
        [ 0.000,  0.001,  1.00]
    ])
if ROI_2D_POLY is None:
    ROI_2D_POLY = np.array([
        [70, 1000],
        [320, 220],
        [940, 220],
        [1120, 900]
    ], dtype=np.int32)

# =========================================================================
# Video Frame Stabilizer (카메라 흔들림 역투영 보정기)
# =========================================================================
class FrameStabilizer:
    def __init__(self, ground_roi=None):
        self.ref_frame = None
        self.ref_kp = None
        self.ref_des = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_H = np.eye(3, dtype=np.float32)
        self.fallback_count = 0
        self.max_fallback = 10
        self.ground_roi = ground_roi

    def initialize(self, frame):
        self.ref_frame = frame.copy()
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.ground_roi is not None:
            cv2.fillPoly(mask, [np.array(self.ground_roi, dtype=np.int32)], 255)
        else:
            mask.fill(255)
            
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(frame, mask)
        self.last_H = np.eye(3, dtype=np.float32)
        self.fallback_count = 0
        print(f"[FrameStabilizer] 기준 프레임(F_0) 초기화 완료. 특징점 수: {len(self.ref_kp) if self.ref_kp else 0}")

    def stabilize_point(self, pt_x, pt_y, frame):
        if self.ref_frame is None:
            self.initialize(frame)
            return (pt_x, pt_y), False

        kp, des = self.orb.detectAndCompute(frame, None)
        low_confidence = False

        if des is not None and self.ref_des is not None and len(kp) > 10:
            try:
                matches = self.bf.match(des, self.ref_des)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = matches[:50]
                
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([self.ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None and np.sum(inliers) >= 4:
                        self.last_H = H
                        self.fallback_count = 0
                    else:
                        self.fallback_count += 1
                        low_confidence = True
                else:
                    self.fallback_count += 1
                    low_confidence = True
            except Exception as match_err:
                print(f"[FrameStabilizer Warning] Match error: {match_err}")
                self.fallback_count += 1
                low_confidence = True
        else:
            self.fallback_count += 1
            low_confidence = True

        if self.fallback_count > self.max_fallback:
            low_confidence = True

        pt = np.array([pt_x, pt_y, 1.0], dtype=np.float32).reshape(3, 1)
        stabilized_pt = np.dot(self.last_H, pt)
        stabilized_pt /= stabilized_pt[2]
        return (float(stabilized_pt[0][0]), float(stabilized_pt[1][0])), low_confidence

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

# 종횡비 동적 타겟 사이즈 결정
is_portrait = height > width
target_size = (720, 1280) if is_portrait else (1280, 720)

# 레터박스 비율 스케일 보정 적용
if ROI_2D_POLY is not None:
    scaled_poly = []
    if width > height:  # 가로 캔버스 레터박스 구조
        content_w = height * (9.0 / 16.0)
        left_black_bar = (width - content_w) / 2.0
        for pt in ROI_2D_POLY:
            mapped_x = left_black_bar + (pt[0] * (content_w / 1080.0))
            mapped_y = pt[1] * (height / 1920.0)
            final_x = mapped_x * (target_size[0] / width)
            final_y = mapped_y * (target_size[1] / height)
            scaled_poly.append([int(final_x), int(final_y)])
    else:  # 순수 세로 동영상 구조
        for pt in ROI_2D_POLY:
            final_x = pt[0] * (target_size[0] / 1080.0)
            final_y = pt[1] * (target_size[1] / 1920.0)
            scaled_poly.append([int(final_x), int(final_y)])
    ROI_2D_POLY = np.array(scaled_poly, dtype=np.int32)

data_list = []
frame_count = 0

# 흔들림 보정기 생성 (보정된 Ground Mask 적용)
stabilizer = FrameStabilizer(ground_roi=ROI_2D_POLY)

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
    if not ret:
        break
        
    # 종횡비 타겟 해상도로 리사이즈
    frame_720p = cv2.resize(frame, target_size)
    
    # 1번 프레임을 stabilizer의 reference frame으로 초기화
    if frame_count == 1:
        stabilizer.initialize(frame_720p)
        
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

    # --- [YOLO11n 기반 가우시안 모자이크 비식별화] ---
    frame_render = frame_720p.copy()
    if yolo_model is not None:
        try:
            results = yolo_model(frame_render, classes=[0], verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    frame_render = apply_mosaic(frame_render, x1, y1, x2, y2)
        except Exception:
            pass

    person_id = 0
    for u, v in peaks:
        person_id += 1
        
        # 흔들림 역투영 보정 적용 (F_0 기준으로 보정, 머리 u, v 기준)
        (su, sv), low_conf = stabilizer.stabilize_point(u, v, frame_720p)
        
        # BEV 3D 공간 미터 좌표 변환 (보정된 발 위치 투영 - sv + 80)
        bev_x, bev_y = transform_pixel_to_bev(su, sv + 80, H_MATRIX)
        
        data_list.append({
            'frame': frame_count,
            'timestamp_sec': round(frame_count / fps, 2),
            'person_id': person_id,
            'pixel_u': round(u, 2),
            'pixel_v': round(v, 2),
            'pixel_foot_u': round(u, 2),
            'pixel_foot_v': round(v + 80, 2),
            'stabilized_u': round(su, 2),
            'stabilized_v': round(sv, 2),
            'bev_x_m': round(bev_x, 2),
            'bev_y_m': round(bev_y, 2),
            'low_confidence': low_conf
        })

        # CSRNet 보행자 중심점 바둑알 마커 시각화 렌더링
        cx, cy = int(u), int(v)
        cv2.circle(frame_render, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(frame_render, (cx, cy), 5, (40, 40, 40), 1)

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