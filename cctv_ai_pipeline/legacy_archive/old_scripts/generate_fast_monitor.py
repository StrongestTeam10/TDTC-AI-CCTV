import os
import re
import sys
import time
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
    parser = argparse.ArgumentParser(description="CCTV FAST Anonymized Monitor Visualization Pipeline")
    parser.add_argument("--input_name", type=str, default="test01", help="test folder input video name")
    parser.add_argument("--peak_threshold", type=float, default=0.005, help="peak detection threshold for CSRNet")
    parser.add_argument("--mosaic_size", type=int, default=15, help="mosaic block scaling divider (larger = blurrier)")
    parser.add_argument("--skip_interval", type=int, default=5, help="frame skip interval for AI inference (larger = faster)")
    parser.add_argument("--ai_width", type=int, default=960, help="CSRNet input width")
    parser.add_argument("--ai_height", type=int, default=540, help="CSRNet input height")
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
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 입력 파일 확인
    input_video_path = os.path.join(TEST_DIR, f"{args.input_name}.mp4")
    if not os.path.exists(input_video_path):
        # uploads 폴더도 확인해 봄 (대시보드 업로드 파일 연동 대비)
        input_video_path = os.path.join(BASE_DIR, "cctv_ai_pipeline", "uploads", f"{args.input_name}.mp4")
        if not os.path.exists(input_video_path):
            # 그냥 절대 경로 통째로 준 경우 가정
            input_video_path = args.input_name
            if not os.path.exists(input_video_path):
                print(f"[Error] Input video file not found: {args.input_name}")
                sys.exit(1)
        
    output_video_path = os.path.join(RESULTS_DIR, f"monitor_fast_{os.path.basename(input_video_path)}")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FAST Monitor] Device: {device}")
    print(f"[FAST Monitor] AI Inference Skip Interval: {args.skip_interval} frames")
    print(f"[FAST Monitor] CSRNet Input Resolution: {args.ai_width}x{args.ai_height}")
    
    # 2. 모델 로드
    # A. CSRNet 로드
    csr_model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
    if not os.path.exists(csr_model_path):
        print(f"[Error] CSRNet model path not found: {csr_model_path}")
        sys.exit(1)
        
    print(f"[FAST Monitor] Loading CSRNet...")
    csr_model = CSRNet().to(device)
    ckpt = torch.load(csr_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    csr_model.load_state_dict(state_dict, strict=False)
    csr_model.eval()
    
    # B. YOLOv8 로드 (초경량 yolov8n 모델 사용)
    if not yolo_available:
        print("[Error] ultralytics library is not installed.")
        sys.exit(1)
        
    # yolov8n.pt 로딩 (없으면 자동 다운로드)
    print(f"[FAST Monitor] Loading Ultra-Lightweight YOLOv8n...")
    yolo_model = YOLO("yolov8n.pt")
    
    # 3. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {input_video_path}")
        sys.exit(1)
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[FAST Monitor] Video Info - Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # VideoWriter 설정 (H264/avc1 시도 후 mp4v 폴백)
    out_monitor = None
    for codec in ['H264', 'avc1', 'mp4v']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_monitor = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if out_monitor.isOpened():
                print(f"[FAST Monitor] VideoWriter 초기화 성공. 사용 코덱: {codec}")
                break
        except Exception:
            pass
            
    if out_monitor is None or not out_monitor.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_monitor = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print("[FAST Monitor] 기본 mp4v 코덱으로 VideoWriter 강제 세팅")
    
    # 이미지 전처리 설정 (CSRNet 용)
    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pbar = tqdm(total=total_frames, desc="Processing FAST Anonymization", unit="frame")
    
    last_yolo_boxes = []
    start_time = time.time()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
            
        # 프레임 스킵 주기에 따른 AI 추론 수행
        if frame_idx == 1 or frame_idx % args.skip_interval == 0:
            try:
                # 1. CSRNet용 해상도 축소 리사이즈 (가속화 적용)
                frame_resized = cv2.resize(frame, (args.ai_width, args.ai_height))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # CSRNet 추론 (밀집도 통계 계산용)
                    csr_model(input_tensor)
                    
                # 2. YOLOv8 Nano 인물 감지 추론 (경량 모델 가속화)
                results = yolo_model(frame, classes=[0], verbose=False)
                last_yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            except Exception as e:
                print(f"\n[AI Error] Frame {frame_idx} Inference failed: {e}")
                
        # 3. 비식별화 모자이크 후처리 렌더링 (매 프레임 고속 처리)
        frame_anonymized = frame.copy()
        for coords in last_yolo_boxes:
            x1, y1, x2, y2 = coords
            frame_anonymized = apply_mosaic(frame_anonymized, x1, y1, x2, y2, neighbor=args.mosaic_size)
            
        # 4. 비디오 프레임 기록
        out_monitor.write(frame_anonymized)
        pbar.update(1)
        
    end_time = time.time()
    elapsed = end_time - start_time
    fps_avg = total_frames / elapsed
    
    # 메모리 정리
    cap.release()
    out_monitor.release()
    pbar.close()
    
    print("\n" + "="*50)
    print("FAST Anonymization Finished Successfully!")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")
    print(f"Average processing speed: {fps_avg:.2f} FPS")
    print(f"Output saved at: {output_video_path}")
    print("="*50)

if __name__ == "__main__":
    main()
