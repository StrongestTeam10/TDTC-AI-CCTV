import os
import re
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import scipy.ndimage as ndimage
import argparse

# YOLOv8 모델 임포트
try:
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# =========================================================================
# 1. CSRNet 모델 클래스 정의
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

def parse_args():
    parser = argparse.ArgumentParser(description="CCTV Anonymized Monitor Visualization Pipeline (YOLO Mosaic + CSRNet Hull)")
    parser.add_argument("--input_name", type=str, default="test01", help="test folder input video name")
    parser.add_argument("--peak_threshold", type=float, default=0.005, help="peak detection threshold for CSRNet")
    parser.add_argument("--mosaic_size", type=int, default=15, help="mosaic block scaling divider (larger = blurrier)")
    return parser.parse_args()

def extract_peaks_from_density(density_map, threshold=0.005):
    """CSRNet 밀도 맵에서 보행자 중심점 Peak (u, v) 좌표 추출"""
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

def apply_mosaic(img, x1, y1, x2, y2, neighbor=15):
    """지정된 바운딩 박스 영역에 픽셀레이션 모자이크 적용"""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return img
        
    roi = img[y1:y2, x1:x2]
    
    # 격자 크기 정의
    grid_w = max(1, w // neighbor)
    grid_h = max(1, h // neighbor)
    
    # 축소 후 확장을 통한 모자이크 효과
    small = cv2.resize(roi, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return img

def main():
    args = parse_args()
    
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DIR = os.path.join(BASE_DIR, "test")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    HEATMAP_DIR = os.path.join(BASE_DIR, "heatmap")
    
    # 입력 파일 확인
    input_video_path = os.path.join(TEST_DIR, f"{args.input_name}.mp4")
    if not os.path.exists(input_video_path):
        print(f"[Error] Input video file not found: {input_video_path}")
        sys.exit(1)
        
    # 파일 이름에서 숫자 번호 추출 (예: test01 -> 01)
    num_match = re.search(r'\d+', args.input_name)
    file_num = num_match.group(0) if num_match else "01"
    
    # 출력 경로 정의 및 폴더 생성
    monitor_out_dir = os.path.join(HEATMAP_DIR, "anonymized")
    os.makedirs(monitor_out_dir, exist_ok=True)
    
    monitor_output_path = os.path.join(monitor_out_dir, f"monitor{file_num}.mp4")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Anonymized Monitor] Device: {device}")
    
    # 2. 모델 로드
    # A. CSRNet 로드
    csr_model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
    if not os.path.exists(csr_model_path):
        print(f"[Error] CSRNet model path not found: {csr_model_path}")
        sys.exit(1)
        
    print(f"[Anonymized Monitor] Loading CSRNet: {csr_model_path}")
    csr_model = CSRNet().to(device)
    ckpt = torch.load(csr_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    csr_model.load_state_dict(state_dict, strict=False)
    csr_model.eval()
    print("[Anonymized Monitor] CSRNet loaded successfully!")
    
    # B. YOLOv8 로드
    if not yolo_available:
        print("[Error] ultralytics library is not installed. YOLO cannot be loaded.")
        sys.exit(1)
        
    yolo_model_path = os.path.join(MODELS_DIR, "bestYOLOm5080model.pt")
    if not os.path.exists(yolo_model_path):
        print(f"[Error] YOLO model path not found: {yolo_model_path}")
        sys.exit(1)
        
    print(f"[Anonymized Monitor] Loading YOLO: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    print("[Anonymized Monitor] YOLO loaded successfully!")
    
    # 3. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {input_video_path}")
        sys.exit(1)
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Anonymized Monitor] Video Info - Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # VideoWriter 설정 (mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_monitor = cv2.VideoWriter(monitor_output_path, fourcc, fps, (width, height))
    
    # 이미지 전처리 설정 (CSRNet 용)
    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pbar = tqdm(total=total_frames, desc="Generating Anonymized Monitor Video", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. 원본 해상도 보존 (배경 시각화용) 및 추론용 크기 조정
        frame_orig = frame.copy()
        frame_resized = cv2.resize(frame_orig, (1280, 720))
        
        # --- [A. 모델 추론 단계 (원본 데이터 기반)] ---
        # A-1. CSRNet 추론
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = csr_model(input_tensor)
            output = torch.clamp(output, min=0)
            
        density_map = output.squeeze().cpu().numpy()
        density_map_resized = cv2.resize(density_map, (width, height))
        
        # CSRNet 기반 보행자 중심점 Peak 추출
        peaks = extract_peaks_from_density(density_map_resized, threshold=args.peak_threshold)
        
        # A-2. YOLOv8 인물 감지 추론
        results = yolo_model(frame_orig, classes=[0], verbose=False)
        boxes = results[0].boxes
        
        # --- [B. 관제 시각화 단계 (비식별화 + 지형물 렌더링)] ---
        # B-1. YOLOv8 감지 영역 실시간 모자이크 씌우기
        if len(boxes) > 0:
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                # 원본 프레임의 인물 구역에 모자이크 적용
                frame_orig = apply_mosaic(frame_orig, x1, y1, x2, y2, neighbor=args.mosaic_size)
                
        # B-2. CSRNet 보행자 밀집 구역 가상 경계 다각형(Convex Hull) 렌더링
        if len(peaks) >= 3:
            pts = np.array([[int(u), int(v)] for u, v in peaks], dtype=np.int32)
            hull = cv2.convexHull(pts)
            
            # 반투명 붉은색 영역용 오버레이
            overlay = frame_orig.copy()
            cv2.drawContours(overlay, [hull], -1, (0, 0, 220), thickness=-1)
            # 20% 투명도로 블렌딩하여 지형지물이 투과되도록 처리
            cv2.addWeighted(overlay, 0.2, frame_orig, 0.8, 0, frame_orig)
            
            # 경계 가이드 라인(오렌지색) 굵게 드로잉
            cv2.polylines(frame_orig, [hull], isClosed=True, color=(0, 150, 255), thickness=3)
            
        # B-3. 개별 보행자 마커(바둑알) 렌더링
        for u, v in peaks:
            cx, cy = int(u), int(v)
            cx = max(0, min(width - 1, cx))
            cy = max(0, min(height - 1, cy))
            
            # 입체감을 주기 위해 안쪽은 흰색 원, 테두리는 어두운 회색으로 2중 마킹
            cv2.circle(frame_orig, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(frame_orig, (cx, cy), 6, (40, 40, 40), 1)
            
        # --- [C. 비디오 프레임 기록] ---
        out_monitor.write(frame_orig)
        
        pbar.update(1)
        
    # 메모리 정리
    cap.release()
    out_monitor.release()
    pbar.close()
    
    print("\n" + "="*50)
    print("Anonymized Monitor Video generation finished successfully!")
    print(f"Output saved at: {monitor_output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
