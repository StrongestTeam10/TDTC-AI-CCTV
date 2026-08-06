# save_CSR_result.py: 입력 비디오에 CSRNet 밀도 예측 모델을 적용하여 
# 사람 밀도 히트맵 및 실시간 인원 카운팅을 원본 영상에 시각화한 동영상(.mp4)을 저장하는 스크립트입니다.
import os
# pyrefly: ignore [missing-import]
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm

# ==========================================
# 1. 경로 및 디바이스(GPU) 설정
# ==========================================
MODEL_PATH = os.environ.get("CSRNET_MODEL_PATH", r"E:\AIVLE_10team\results\csrnet_ultimate_epoch_8.pth")
VIDEO_PATH = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_cafe_output.mp4")
SAVE_OUTPUT_PATH = os.environ.get("CSR_RESULT_MP4", r"E:\test\cctv_cafe_result.mp4")

# 🔥 GPU 사용 가능 여부 강제 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 GPU 가속 활성화 완료! ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("⚠️ WARNING: GPU(CUDA)를 찾을 수 없어 CPU로 실행됩니다.")
    print("   (PyTorch CUDA 버전이 올바르게 설치되어 있는지 환경을 확인해 주세요!)")

# ==========================================
# 2. CSRNet 모델 클래스 정의 (석훈님 모델 구조)
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
        x = self.frontend(x)
        x = self.backend(x)
        return x

# ==========================================
# 3. 모델 로드
# ==========================================
print(f"📂 CSRNet 가중치 로드 중...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")
    exit()

model = CSRNet().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print("✅ 모델 로드 완료!\n")

# ==========================================
# 4. 이미지 전처리 & 720p (1280x720) 비디오 설정
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    print("❌ 비디오 파일을 열 수 없습니다!")
    if temp_read_path and os.path.exists(temp_read_path):
        os.remove(temp_read_path)
    exit()

# 🔥 720p 해상도 고정 설정 (가로 1280, 세로 720)
TARGET_W, TARGET_H = 1280, 720

fps          = cap.get(cv2.CAP_PROP_FPS) or 15.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out, temp_write_path = get_safe_video_writer(SAVE_OUTPUT_PATH, fourcc, fps, (TARGET_W, TARGET_H))

print(f"🎬 720p({TARGET_W}x{TARGET_H}) 해상도 최적화 추론 시작! (총 {total_frames} 프레임)\n")

# ==========================================
# 5. 진행률 바(tqdm) 적용 추론 루프
# ==========================================
pbar = tqdm(total=total_frames, desc="🎬 CSRNet 720p 추론 진행 중", unit="frame")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 720p 해상도로 리사이즈
    frame_720p = cv2.resize(frame, (TARGET_W, TARGET_H))
    
    img_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, min=0)
        pred_count = output.sum().item()

    # 히트맵 오버레이 처리
    density_map = output.squeeze().cpu().numpy()
    density_map_resized = cv2.resize(density_map, (TARGET_W, TARGET_H))

    max_val = density_map_resized.max()
    if max_val > 0:
        density_norm = (density_map_resized / max_val * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map_resized, dtype=np.uint8)

    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    
    # 720p 프레임 + 히트맵 오버레이
    result_frame = cv2.addWeighted(frame_720p, 0.5, heatmap, 0.5, 0)

    # 텍스트 오버레이
    text = f"Estimated Crowd: {int(np.round(pred_count))} people"
    cv2.putText(result_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out.write(result_frame)
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
    if os.path.exists(SAVE_OUTPUT_PATH):
        os.remove(SAVE_OUTPUT_PATH)
    shutil.move(temp_write_path, SAVE_OUTPUT_PATH)

print("\n" + "="*50)
print("🎉 720p CSRNet 추론 및 결과 영상 저장 완료!")
print(f"📁 저장 파일: {SAVE_OUTPUT_PATH}")
print("="*50)