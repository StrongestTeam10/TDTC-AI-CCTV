import os
import re
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description="CCTV Heatmap Generation Pipeline (CSRNet, YOLO, Combined)")
    parser.add_argument("--input_name", type=str, default="test01", help="test folder input video name")
    parser.add_argument("--threshold", type=int, default=15, help="noise threshold (0~255)")
    return parser.parse_args()

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
    csr_out_dir = os.path.join(HEATMAP_DIR, "csrnet")
    yolo_out_dir = os.path.join(HEATMAP_DIR, "yolo")
    csryo_out_dir = os.path.join(HEATMAP_DIR, "csryo")
    
    os.makedirs(csr_out_dir, exist_ok=True)
    os.makedirs(yolo_out_dir, exist_ok=True)
    os.makedirs(csryo_out_dir, exist_ok=True)
    
    csr_output_path = os.path.join(csr_out_dir, f"csrnet{file_num}.mp4")
    yolo_output_path = os.path.join(yolo_out_dir, f"yolo{file_num}.mp4")
    csryo_output_path = os.path.join(csryo_out_dir, f"csryo{file_num}.mp4")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Heatmap Pipeline] Device: {device}")
    
    # 2. 모델 로드
    # A. CSRNet 로드
    csr_model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
    if not os.path.exists(csr_model_path):
        print(f"[Error] CSRNet model path not found: {csr_model_path}")
        sys.exit(1)
        
    print(f"[Heatmap Pipeline] Loading CSRNet: {csr_model_path}")
    csr_model = CSRNet().to(device)
    ckpt = torch.load(csr_model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    csr_model.load_state_dict(state_dict, strict=False)
    csr_model.eval()
    print("[Heatmap Pipeline] CSRNet loaded successfully!")
    
    # B. YOLOv8 로드
    if not yolo_available:
        print("[Error] ultralytics library is not installed. YOLO heatmap cannot be created.")
        sys.exit(1)
        
    yolo_model_path = os.path.join(MODELS_DIR, "bestYOLOm5080model.pt")
    if not os.path.exists(yolo_model_path):
        print(f"[Error] YOLO model path not found: {yolo_model_path}")
        sys.exit(1)
        
    print(f"[Heatmap Pipeline] Loading YOLO: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    print("[Heatmap Pipeline] YOLO loaded successfully!")
    
    # 3. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {input_video_path}")
        sys.exit(1)
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Heatmap Pipeline] Video Info - Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # VideoWriter 설정 (mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_csr = cv2.VideoWriter(csr_output_path, fourcc, fps, (width, height))
    out_yolo = cv2.VideoWriter(yolo_output_path, fourcc, fps, (width, height))
    out_csryo = cv2.VideoWriter(csryo_output_path, fourcc, fps, (width, height))
    
    # 이미지 전처리 설정 (CSRNet 용)
    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 히트맵 마스킹 헬퍼 함수
    def create_masked_heatmap(norm_img, threshold):
        color_map = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        mask = norm_img > threshold
        color_map[~mask] = 0
        return color_map
        
    pbar = tqdm(total=total_frames, desc="Generating Heatmaps", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1280x720 해상도 정규화 (모델 추론용)
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # --- [A. CSRNet 추론 및 히트맵 생성] ---
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = csr_model(input_tensor)
            output = torch.clamp(output, min=0)
            
        density_map = output.squeeze().cpu().numpy()
        density_map_resized = cv2.resize(density_map, (width, height))
        
        # CSRNet 밀도 맵 정규화 (0 ~ 255)
        if density_map_resized.max() > 0:
            density_csr_normalized = cv2.normalize(density_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            density_csr_normalized = np.zeros((height, width), dtype=np.uint8)
            
        # --- [B. YOLOv8 추론 및 히트맵 생성] ---
        # 사람 클래스(0) 검출
        results = yolo_model(frame, classes=[0], verbose=False)
        yolo_density = np.zeros((height, width), dtype=np.float32)
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                
                # 보행자 몸통 중심점 좌표 계산
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                cx = max(0, min(width - 1, cx))
                cy = max(0, min(height - 1, cy))
                
                cv2.circle(yolo_density, (cx, cy), 15, 1.0, -1)
                
            # 가우시안 블러 적용하여 부드러운 가상 밀집 맵 생성
            yolo_density_blur = cv2.GaussianBlur(yolo_density, (101, 101), 35)
            
            # YOLO 밀도 맵 정규화 (0 ~ 255)
            if yolo_density_blur.max() > 0:
                density_yolo_normalized = cv2.normalize(yolo_density_blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                density_yolo_normalized = np.zeros((height, width), dtype=np.uint8)
        else:
            density_yolo_normalized = np.zeros((height, width), dtype=np.uint8)
            
        # --- [C. 두 모델 밀도 맵 병합 (csryo)] ---
        density_combined = cv2.addWeighted(density_csr_normalized, 0.5, density_yolo_normalized, 0.5, 0)
        
        # --- [D. 컬러맵 입히기 및 노이즈 마스킹 (비식별화 적용)] ---
        heatmap_csr = create_masked_heatmap(density_csr_normalized, args.threshold)
        heatmap_yolo = create_masked_heatmap(density_yolo_normalized, args.threshold)
        heatmap_combined = create_masked_heatmap(density_combined, args.threshold)
        
        # --- [E. 비디오 프레임 기록] ---
        out_csr.write(heatmap_csr)
        out_yolo.write(heatmap_yolo)
        out_csryo.write(heatmap_combined)
        
        pbar.update(1)
        
    # 메모리 정리
    cap.release()
    out_csr.release()
    out_yolo.release()
    out_csryo.release()
    pbar.close()
    
    print("\n" + "="*50)
    print("Heatmap generation finished successfully!")
    print(f"CSRNet output: {csr_output_path}")
    print(f"YOLO output:   {yolo_output_path}")
    print(f"Combined (csryo) output: {csryo_output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
