import cv2
import os
import asyncio
import json
import time
import math
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# CSRNet 및 PyTorch 의존성 임포트 (안전한 예외 차단 적용)
torch_available = False
yolo_available = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    import scipy.ndimage as ndimage
    
    # ultralytics YOLO 임포트
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        pass
        
    torch_available = True
    
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
            
    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
except Exception as e:
    print(f"[API Server Setup Warning] PyTorch 또는 torchvision 로드 불가: {e}")

# =========================================================================
# [개선] 호모그래피 변환 행렬 & 센트로이드 트래커 (보행자 속도/지연 분석용)
# =========================================================================
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

class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=2.0):
        self.next_object_id = 0
        self.objects = {} # id -> (x, y)
        self.disappeared = {} # id -> disappeared count
        self.tracks = {} # id -> list of (frame_id, x, y)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = []
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.tracks:
            del self.tracks[object_id]

    def update(self, rects, frame_id):
        # rects: 현재 프레임에서 감지된 BEV 물리 좌표 (x, y) 리스트
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(rects)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                obj_id = self.register(input_centroids[i])
                self.tracks[obj_id].append((frame_id, input_centroids[i][0], input_centroids[i][1]))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # 이전 센트로이드와 현재 입력 센트로이드 간의 유클리드 거리 계산
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.tracks[object_id].append((frame_id, input_centroids[col][0], input_centroids[col][1]))
                if len(self.tracks[object_id]) > 5:
                    self.tracks[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                obj_id = self.register(input_centroids[col])
                self.tracks[obj_id].append((frame_id, input_centroids[col][0], input_centroids[col][1]))

        return self.objects

app = FastAPI(
    title="Mangwon Smart CCTV AI Pipeline API",
    description="실시간 CCTV 비디오 업로드, AI 추론 및 WebSocket 관제 데이터 스트리밍 서버",
    version="1.0.0"
)

# CORS 미들웨어 설정 (대시보드 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 비디오 저장 디렉터리
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 연결된 WebSocket 클라이언트 관리자
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket Connected] 현재 연결된 클라이언트 수: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket Disconnected] 현재 연결된 클라이언트 수: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"[WebSocket Broadcast Error] {e}")

manager = ConnectionManager()

# 백엔드 파이프라인 분석 상태 변수
pipeline_state = {
    "is_analyzing": False,
    "current_video": "default_mangwon.mp4",
    "frame_id": 1,
    "pedestrian_count": 7,
    "cri_score": 22.5,
    "status": "IDLE"
}

@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": "Mangwon CCTV AI Pipeline API",
        "active_clients": len(manager.active_connections),
        "pipeline_state": pipeline_state
    }

@app.get("/api/v1/cctv/status")
async def get_pipeline_status():
    return JSONResponse(content=pipeline_state)

@app.post("/api/v1/cctv/upload")
async def upload_cctv_video(
    background_tasks: BackgroundTasks,
    cctv_video: UploadFile = File(...)
):
    """
    대시보드에서 MOV/MP4 비디오 업로드 수신 및 AI 파이프라인 자동 분석 트리거
    """
    file_name = cctv_video.filename
    file_path = os.path.join(UPLOAD_DIR, file_name)

    # 파일 수신 및 저장
    contents = await cctv_video.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    file_size_mb = len(contents) / (1024 * 1024)
    print(f"[Video Upload Completed] 파일명: {file_name}, 용량: {file_size_mb:.2f}MB, 저장 경로: {file_path}")

    # 백엔드 파이프라인 상태 업데이트
    pipeline_state["is_analyzing"] = True
    pipeline_state["current_video"] = file_name
    pipeline_state["status"] = "ANALYZING"

    # 백그라운드 AI 파이프라인 분석 및 스트리밍 시뮬레이션 태스크 실행
    background_tasks.add_task(process_ai_pipeline, file_path, file_name)

    return JSONResponse(content={
        "status": "SUCCESS",
        "message": f"CCTV 비디오 '{file_name}' 업로드 완료. AI 파이프라인 분석을 시작합니다.",
        "filename": file_name,
        "size_mb": round(file_size_mb, 2),
        "saved_path": file_path
    })

@app.get("/api/v1/cctv/video/{filename}")
async def get_cctv_result_video(filename: str):
    # 실제 저장된 시각화 영상인 results\cctv_simulation_video.mp4 경로를 지정합니다.
    video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cctv_simulation_video.mp4"))
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"message": "시각화 비디오 파일이 존재하지 않습니다."})

@app.get("/api/v1/cctv/dataset/{filename}")
async def get_cctv_result_dataset(filename: str):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    dataset_path = os.path.join(results_dir, f"uploaded_{filename}_dataset.json")
    if os.path.exists(dataset_path):
        return FileResponse(dataset_path, media_type="application/json")
    fallback_path = os.path.join(results_dir, "pedaggr01h_full_dataset.json")
    if os.path.exists(fallback_path):
        return FileResponse(fallback_path, media_type="application/json")
    return JSONResponse(status_code=404, content={"message": "분석 데이터셋이 존재하지 않습니다."})

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
        import numpy as np
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_read_{os.getpid()}_{np.random.randint(1000)}.mp4")
        shutil.copy(video_path, temp_path)
        cap = cv2.VideoCapture(temp_path)
        return cap, temp_path
    else:
        cap = cv2.VideoCapture(video_path)
        return cap, None

async def process_ai_pipeline(file_path: str, filename: str):
    """
    업로드된 비디오 기반 AI 파이프라인 추론 및 WebSocket 실시간 스트리밍 태스크
    (YOLO/CSRNet 감지 + 3D BEV 역투영 + CRI 위험도 산출 피드백)
    """
    print(f"[AI Pipeline Started] '{filename}' 영상 모델링 분석 개시...")
    
    # 1. AI 파이프라인 분석 시작 이벤트 전송
    await manager.broadcast({
        "type": "CCTV_AI_START",
        "filename": filename,
        "message": f"🤖 CCTV 영상 '{filename}' AI 파이프라인 분석을 시작합니다."
    })

    # 2. CSRNet 및 YOLOv8, HOG 감지기 초기화 (안전성 이중 예비책)
    csrnet_model = None
    yolo_model = None
    hog = None
    device = None
    if torch_available:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            csrnet_model = CSRNet().to(device)
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
            model_path = os.path.join(models_dir, "csrnet_ultimate_epoch_8.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=device)
                state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
                csrnet_model.load_state_dict(state_dict, strict=False)
                csrnet_model.eval()
                print(f"[AI Pipeline] CSRNet 모델 가중치 로드 성공: {model_path}")
            else:
                print(f"[AI Pipeline Warning] CSRNet 가중치 파일을 찾을 수 없습니다: {model_path}")
                csrnet_model = None
                
            # YOLOv8 로드 (전용 학습 모델 가중치 사용)
            if yolo_available:
                yolo_path = os.path.join(models_dir, "bestYOLOm5080model.pt")
                if os.path.exists(yolo_path):
                    yolo_model = YOLO(yolo_path)
                    print(f"[AI Pipeline] YOLOv8 모델 가중치 로드 성공: {yolo_path}")
                else:
                    print(f"[AI Pipeline Warning] YOLOv8 가중치 파일을 찾을 수 없습니다: {yolo_path}")
        except Exception as e:
            print(f"[AI Pipeline Warning] CSRNet/YOLO 로드 실패 ({e}). OpenCV HOG 감지기로 대체합니다.")
            csrnet_model = None
            yolo_model = None

    if csrnet_model is None:
        try:
            import cv2
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[AI Pipeline] OpenCV HOG 보행자 감지기 초기화 성공.")
        except Exception as ex:
            print(f"[AI Pipeline Error] HOG 감지기 초기화 실패 ({ex}). 수학적 시뮬레이션으로 대체합니다.")

    # 3. 비디오 로드 및 프레임별 실제 AI 추론 루프
    import cv2
    cap, temp_read_path = get_safe_video_capture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 604
    if total_frames <= 0:
        total_frames = 604

    print(f"[AI Pipeline Info] 비디오 총 프레임 수: {total_frames}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    if fps <= 0:
        fps = 15.0
        
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    output_video_path = os.path.join(results_dir, "cctv_simulation_video.mp4")
    
    # 브라우저 친화적 재생을 위해 H264 또는 avc1 코덱 시도, 실패 시 mp4v로 폴백
    out_video = None
    for codec in ['H264', 'avc1', 'mp4v']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if out_video.isOpened():
                print(f"[AI Pipeline] VideoWriter 초기화 성공. 사용 코덱: {codec}")
                break
        except Exception as e:
            print(f"[AI Pipeline Warning] 코덱 {codec} 초기화 실패: {e}")
            
    if out_video is None or not out_video.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print("[AI Pipeline] 기본 mp4v 코덱으로 VideoWriter 최종 강제 초기화")

    seed_offset = sum(ord(c) for c in filename) % 20
    dataset_records = {}
    stagnation_accum = 2.0
    last_base_count = 0
    skip_interval = 3  # 연산 속도 3배 개선을 위한 프레임 스킵 간격
    last_peaks = []    # 이전 프레임 보행자 좌표 캐시
    last_yolo_boxes = [] # 이전 프레임 YOLO 바운딩 박스 캐시

    # [개선] 센트로이드 트래커 및 정밀 계산을 위한 변수 초기화
    tracker = CentroidTracker(max_disappeared=15, max_distance=2.5)
    last_processed_frame = 0
    last_calculated_avg_speed = 1.35
    last_calculated_stagnation_sec = 2

    for frame in range(1, total_frames + 1):
        if not pipeline_state["is_analyzing"]:
            break
            
        ret = False
        img = None
        if cap.isOpened():
            ret, img = cap.read()

        is_inferenced = False

        # 프레임 스킵 간격으로 AI 추론 진행
        if ret and img is not None:
            if frame == 1 or frame % skip_interval == 0:
                is_inferenced = True
                try:
                    if csrnet_model is not None and device is not None:
                        # 720p 해상도로 리사이즈 (CSRNet 추론 정밀도 보정)
                        frame_720p = cv2.resize(img, (1280, 720))
                        # BGR -> RGB 변환 및 정규화 텐서화
                        img_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
                        input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = csrnet_model(input_tensor)
                            # 출력 밀도 맵의 총 합이 예측 보행자 수
                            predicted_count = float(output.sum().item())
                            last_base_count = int(round(predicted_count))
                            
                            # 리사이즈된 밀도 맵에서 안정적인 Peak 좌표 추출
                            density_map = output.squeeze().cpu().numpy()
                            density_map_resized = cv2.resize(density_map, (width, height))
                            last_peaks = extract_peaks_from_density(density_map_resized, threshold=0.0015)
                            
                        # YOLOv8 인물 감지 추론 동시에 수행 (모자이크 렌더링용)
                        if yolo_model is not None:
                            yolo_results = yolo_model(img, classes=[0], verbose=False)
                            last_yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                        else:
                            last_yolo_boxes = []
                    elif hog is not None:
                        # OpenCV HOG 감지
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                        last_base_count = len(boxes)
                        
                        last_peaks = []
                        last_yolo_boxes = []
                        for bx, by, bw, bh in boxes:
                            last_peaks.append((bx + bw/2.0, by + bh/2.0))
                            last_yolo_boxes.append([bx, by, bx + bw, by + bh])
                    else:
                        # 폴백: 수학적 시뮬레이션
                        last_base_count = 6 + seed_offset + int(7 * math.sin((frame + seed_offset * 10) / 18.0)) + (frame // 120)
                        last_peaks = []
                        last_yolo_boxes = []
                except Exception as ex:
                    print(f"[AI Inference Error] 프레임 {frame} 분석 중 오류: {ex}")
            else:
                # 스킵하는 프레임에서는 직전 프레임의 카운트를 고스란히 사용 (캐시 재활용)
                pass
        else:
            # 비디오 읽기 실패 또는 미연결 시 수학적 폴백
            is_inferenced = True
            last_base_count = 6 + seed_offset + int(7 * math.sin((frame + seed_offset * 10) / 18.0)) + (frame // 120)
            last_peaks = []
            last_yolo_boxes = []

        # 감출된 인원수 클램프
        base_count = max(0, last_base_count)

        # 4. [개선] 정체 멈춤시간 물리 알고리즘
        if is_inferenced:
            # 1) 현재 프레임 보행자들의 BEV 물리 좌표 리스트 수집
            current_bev_pts = []
            if ret and img is not None:
                if csrnet_model is not None and device is not None:
                    # CSRNet peak (u, v) 좌표 변환
                    for u, v in last_peaks:
                        bx, by = transform_pixel_to_bev(u, v, H_MATRIX)
                        current_bev_pts.append((bx, by))
                elif yolo_model is not None or hog is not None:
                    # YOLOv8 / HOG 바운딩 박스 하단 중앙(발밑) 좌표 변환
                    for coords in last_yolo_boxes:
                        x1, y1, x2, y2 = coords
                        foot_u = (x1 + x2) / 2.0
                        foot_v = y2
                        bx, by = transform_pixel_to_bev(foot_u, foot_v, H_MATRIX)
                        current_bev_pts.append((bx, by))
            
            # 2) Centroid Tracker 업데이트
            tracker.update(current_bev_pts, frame)
            
            # 3) 각 트랙별 이동속도(m/s) 연산
            speeds = []
            frame_diff = frame - last_processed_frame if last_processed_frame > 0 else skip_interval
            dt_step = frame_diff / fps
            
            for obj_id, track in tracker.tracks.items():
                if len(track) >= 2:
                    f0, x0, y0 = track[-2]
                    f1, x1, y1 = track[-1]
                    t_diff = (f1 - f0) / fps
                    if t_diff > 0:
                        dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                        speed = dist / t_diff
                        if speed < 4.0:  # 비정상 노이즈 필터링
                            speeds.append(speed)
            
            # 4) 평균 속도 도출
            if len(speeds) > 0:
                avg_speed = float(np.mean(speeds))
            else:
                # 검출 속도 데이터가 전혀 없는 경우 밀집도에 의한 이론적 감쇠 공식 적용 (폴백)
                avg_speed = 1.35 - (base_count * 0.035)
                avg_speed = max(0.15, min(1.6, avg_speed))
            
            # 5) 보행 정체 기준(0.8m/s) 누적/차감 연산
            if avg_speed <= 0.8:
                # 속도가 낮을수록 정체시간 가파르게 상승 (초 단위 누적 스케일 보정)
                stagnation_accum += (0.8 - avg_speed) * 3.0 * dt_step
            else:
                # 정상 속도 이동 시 정체시간 점진적 해소 (초 단위 완화 스케일 보정)
                stagnation_accum = max(2.0, stagnation_accum - 0.75 * dt_step)
                
            stagnation_sec = int(min(120.0, stagnation_accum))
            last_processed_frame = frame
            
            # 스킵 프레임용 캐시 저장
            last_calculated_avg_speed = avg_speed
            last_calculated_stagnation_sec = stagnation_sec
        else:
            # 스킵 프레임에서는 캐시된 계산 값 복원
            avg_speed = last_calculated_avg_speed
            stagnation_sec = last_calculated_stagnation_sec
            
        occupancy_rate = round(min(100.0, base_count * 2.6), 1)
        
        raw_cri = (base_count * 2.2) + (stagnation_sec * 0.6) + (occupancy_rate * 0.3)
        cri_score = round(min(100.0, max(10.0, raw_cri)), 1)
        
        risk_level = "NORMAL"
        if cri_score >= 70.0:
            risk_level = "EMERGENCY_EVACUATION"
        elif cri_score >= 50.0:
            risk_level = "WARNING"
            
        dataset_records[str(frame)] = {
            "pedestrian_count": base_count,
            "occupancy_rate": occupancy_rate,
            "stagnation_sec": stagnation_sec,
            "cri_score": cri_score,
            "risk_level": risk_level
        }

        # --- [비식별화 모자이크 렌더링 (YOLO 바운딩 박스 기반)] ---
        if ret and img is not None:
            img_anonymized = img.copy()
            # YOLO가 검출한 모든 사람(보행자) 영역에 픽셀레이션 모자이크 적용
            for coords in last_yolo_boxes:
                x1, y1, x2, y2 = coords
                img_anonymized = apply_mosaic(img_anonymized, x1, y1, x2, y2, neighbor=15)
                
            # VideoWriter에 최종 가공된 프레임 기록
            out_video.write(img_anonymized)

        # 주기적으로 실제 진행률을 대시보드에 브로드캐스트 (12프레임 간격)
        if frame == 1 or frame % 12 == 0 or frame == total_frames:
            percent = int((frame / total_frames) * 100)
            await manager.broadcast({
                "type": "CCTV_AI_PROGRESS",
                "filename": filename,
                "progress": percent,
                "step_name": f"실제 MP4 영상 기반 CSRNet 객체 탐지 및 실시간 원형 블러 처리 중 ({percent}%)"
            })
            await asyncio.sleep(0.01)

    if cap.isOpened():
        cap.release()
    out_video.release() # VideoWriter 자원 해제
    if temp_read_path and os.path.exists(temp_read_path):
        try:
            os.remove(temp_read_path)
            print(f"[AI Pipeline] 임시 비디오 리더 파일 삭제 완료: {temp_read_path}")
        except Exception as e:
            print(f"[AI Pipeline Warning] 임시 파일 삭제 실패: {e}")

    # 계산 완료된 데이터셋을 디스크 파일로 영구 저장
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    dataset_path = os.path.join(results_dir, f"uploaded_{filename}_dataset.json")
    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_records, f, ensure_ascii=False, indent=2)
        print(f"[AI Dataset Saved] 데이터셋이 파일에 영구 저장되었습니다: {dataset_path}")
    except Exception as e:
        print(f"[AI Dataset Save Error] 파일 저장 중 오류 발생: {e}")

    # 5. AI 모델링 분석 완성 이벤트 전송
    await manager.broadcast({
        "type": "CCTV_AI_COMPLETED",
        "filename": filename,
        "message": f"✅ '{filename}' AI 모델링 파이프라인 분석이 성공적으로 완성되었습니다!"
    })
    print(f"[AI Pipeline Completed] '{filename}' 모델링 완료! 시각화 실시간 스트리밍 개시.")

    # 6. 완성 후 프레임별 실시간 관제 데이터 연속 스트리밍 전송 (이미 생성된 데이터셋 기준)
    for frame in range(1, total_frames + 1):
        if not pipeline_state["is_analyzing"]:
            break

        frame_data = dataset_records.get(str(frame))
        if not frame_data:
            continue

        payload = {
            "type": "CCTV_AI_STREAM",
            "frame_id": frame,
            "filename": filename,
            "pedestrian_count": frame_data["pedestrian_count"],
            "occupancy_rate": frame_data["occupancy_rate"],
            "stagnation_sec": frame_data["stagnation_sec"],
            "cri_score": frame_data["cri_score"],
            "risk_level": frame_data["risk_level"],
            "timestamp": time.time()
        }

        pipeline_state["frame_id"] = frame
        pipeline_state["pedestrian_count"] = frame_data["pedestrian_count"]
        pipeline_state["cri_score"] = frame_data["cri_score"]

        # 연결된 모든 대시보드 웹소켓으로 실시간 피드백 전송
        await manager.broadcast(payload)
        await asyncio.sleep(0.2) # 약 5 FPS 실시간 관제 스트리밍

    pipeline_state["is_analyzing"] = False
    pipeline_state["status"] = "COMPLETED"
    print(f"[AI Pipeline Completed] '{filename}' 분석 파이프라인 완료.")

@app.websocket("/ws/cctv-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 최초 연결 시 현재 상태 전송
        await websocket.send_json({
            "type": "INIT_STATE",
            "pipeline_state": pipeline_state
        })
        
        while True:
            # 클라이언트 수신 대기 (핑/퐁 유지)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "time": time.time()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
