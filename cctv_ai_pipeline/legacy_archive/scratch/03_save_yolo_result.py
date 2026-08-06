# save_yolo_result.py: 입력 비디오에 YOLOv8 검출 모델을 적용하여 사람 바운딩 박스를 시각화한 결과 동영상(.mp4)을 생성 및 저장하는 스크립트입니다.
import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import shutil

# 💡 한글 경로 지원용 비디오 입출력 헬퍼
def get_safe_video_capture(video_path):
    is_unicode = False
    try:
        video_path.encode('ascii')
    except UnicodeEncodeError:
        is_unicode = True
        
    if is_unicode and os.path.exists(video_path):
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

# 1. 경로 설정
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", r"E:\AIVLE_10team\results\bestYOLOm5080model.pt")
VIDEO_PATH = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_EXCO_test_output.mp4")
SAVE_OUTPUT_PATH = os.environ.get("YOLO_RESULT_MP4", r"E:\test\cctv_EXCO_yolo_result.mp4")

# 2. 모델 로드
print("📂 YOLO 모델 로드 중...")
model = YOLO(MODEL_PATH)

# 3. 비디오 파일 열기
cap, temp_read_path = get_safe_video_capture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ 비디오 파일을 열 수 없습니다. 경로를 확인해주세요!")
    if temp_read_path and os.path.exists(temp_read_path):
        os.remove(temp_read_path)
    exit()

# 원본 영상의 비디오 정보(해상도, FPS) 가져오기
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): 
    fps = 15.0 # FPS 예외 처리

# 🔥 4. 결과 영상을 저장할 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out, temp_write_path = get_safe_video_writer(SAVE_OUTPUT_PATH, fourcc, fps, (width, height))

if out is None or not out.isOpened():
    print("❌ 비디오 저장 라이터를 열 수 없습니다!")
    cap.release()
    if temp_read_path and os.path.exists(temp_read_path):
        os.remove(temp_read_path)
    exit()

print(f"🎬 동영상 YOLO 추론 및 저장 시작!")
print(f"💾 결과 저장 경로: {SAVE_OUTPUT_PATH}\n")

confidences = []
total_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    total_frames += 1
    
    # YOLO 추론 실행 (ByteTrack 적용)
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            confidences.extend(confs)
            
            # 박스 및 ID가 그려진 프레임 이미지
            annotated_frame = r.plot()
        else:
            annotated_frame = frame
 
    # 🔥 5. 결과 프레임을 비디오 파일에 쓰기 (저장)
    out.write(annotated_frame)

    # 실시간 화면 출력 (원치 않으면 아래 2줄을 주석처리하면 더 빠르게 저장돼!)
    cv2.imshow("YOLO11s + ByteTrack Test", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release() # 🔥 비디오 파일 저장 완료 처리
cv2.destroyAllWindows()

# 임시 입력 파일 정리
if temp_read_path and os.path.exists(temp_read_path):
    try:
        os.remove(temp_read_path)
    except Exception:
        pass

# 임시 출력 파일 목적지로 이동
if temp_write_path and os.path.exists(temp_write_path):
    if os.path.exists(SAVE_OUTPUT_PATH):
        os.remove(SAVE_OUTPUT_PATH)
    shutil.move(temp_write_path, SAVE_OUTPUT_PATH)

# ==========================================
# 📊 결과 출력
# ==========================================
print("\n" + "="*40)
print("🎉 결과 영상 저장 완료!")
print("="*40)
print(f"📁 저장된 파일: {SAVE_OUTPUT_PATH}")
if confidences:
    avg_conf = np.mean(confidences) * 100
    print(f"🎯 평균 인식 신뢰도(Confidence): {avg_conf:.2f}%")
print("="*40)