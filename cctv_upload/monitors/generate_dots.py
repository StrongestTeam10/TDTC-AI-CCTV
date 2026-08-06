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
    parser = argparse.ArgumentParser(description="CCTV 2D Dots & Virtual Border Generation Pipeline (CSRNet)")
    parser.add_argument("--input_name", type=str, default="test01", help="test folder input video name")
    parser.add_argument("--peak_threshold", type=float, default=0.005, help="peak detection threshold for CSRNet")
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
    csrdots_out_dir = os.path.join(HEATMAP_DIR, "csrdots")
    os.makedirs(csrdots_out_dir, exist_ok=True)
    
    csrdots_output_path = os.path.join(csrdots_out_dir, f"csrdots{file_num}.mp4")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Dots & Border Pipeline] Device: {device}")
    
    # 2. 모델 로드
    csr_model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
    if not os.path.exists(csr_model_path):
        print(f"[Error] CSRNet model path not found: {csr_model_path}")
        sys.exit(1)
        
    print(f"[Dots & Border Pipeline] Loading CSRNet: {csr_model_path}")
    csr_model = CSRNet().to(device)
    ckpt = torch.load(csr_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    csr_model.load_state_dict(state_dict, strict=False)
    csr_model.eval()
    print("[Dots & Border Pipeline] CSRNet loaded successfully!")
    
    # 3. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {input_video_path}")
        sys.exit(1)
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Dots & Border Pipeline] Video Info - Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # VideoWriter 설정 (mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_csrdots = cv2.VideoWriter(csrdots_output_path, fourcc, fps, (width, height))
    
    # 이미지 전처리 설정 (CSRNet 용)
    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pbar = tqdm(total=total_frames, desc="Generating Dots & Lines", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1280x720 해상도 정규화 (모델 추론용)
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # --- [A. CSRNet 추론 및 밀도 맵 생성] ---
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = csr_model(input_tensor)
            output = torch.clamp(output, min=0)
            
        density_map = output.squeeze().cpu().numpy()
        density_map_resized = cv2.resize(density_map, (width, height))
        
        # --- [B. CSRNet 좌표 기반 2D 바둑알 및 가상 경계 다각형(Convex Hull) 생성] ---
        # CSRNet 밀도 히트맵 상의 2D 인파 좌표 Peak 추출
        peaks = extract_peaks_from_density(density_map_resized, threshold=args.peak_threshold)
        
        # 검은색 배경 캔버스 생성 (비식별화)
        dots_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 보행자가 3명 이상 있는 경우, 이들을 모두 감싸는 가상의 반투명 경계 구역(Convex Hull)을 렌더링
        if len(peaks) >= 3:
            pts = np.array([[int(u), int(v)] for u, v in peaks], dtype=np.int32)
            hull = cv2.convexHull(pts)
            
            # 반투명 영역용 오버레이 프레임 생성
            overlay = dots_canvas.copy()
            # 껍질 내부 채우기 (연한 붉은색)
            cv2.drawContours(overlay, [hull], -1, (0, 0, 200), thickness=-1)
            # 알파 채널 블렌딩 적용 (30% 반투명도)
            cv2.addWeighted(overlay, 0.3, dots_canvas, 0.7, 0, dots_canvas)
            
            # 다각형 테두리 그리기 (오렌지/노란색 경계 가이드 라인)
            cv2.polylines(dots_canvas, [hull], isClosed=True, color=(0, 180, 255), thickness=2)
            
        # 개별 보행자 자리에 2D 바둑알 그리기 (입체감 있는 테두리 포함)
        for u, v in peaks:
            cx, cy = int(u), int(v)
            cx = max(0, min(width - 1, cx))
            cy = max(0, min(height - 1, cy))
            
            # 흰색 바둑알 본체
            cv2.circle(dots_canvas, (cx, cy), 7, (255, 255, 255), -1)
            # 검은색 아웃라인 테두리선 추가로 바둑알 디자인 강화
            cv2.circle(dots_canvas, (cx, cy), 7, (50, 50, 50), 1)
            
        # --- [C. 비디오 프레임 기록] ---
        out_csrdots.write(dots_canvas)
        
        pbar.update(1)
        
    # 메모리 정리
    cap.release()
    out_csrdots.release()
    pbar.close()
    
    print("\n" + "="*50)
    print("Dots & Virtual Border generation finished successfully!")
    print(f"Output saved at: {csrdots_output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
