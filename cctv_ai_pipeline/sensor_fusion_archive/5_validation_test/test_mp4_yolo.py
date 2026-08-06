# test_mp4_yolo.py: YOLOv8 모델과 ByteTrack을 사용하여 입력 비디오 프레임별 실시간 객체 트래킹 결과를 화면에 시각화하고 
# 평균 정확도를 측정하는 테스트 스크립트입니다.
import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import shutil

# 💡 한글 경로 지원용 비디오 로드 헬퍼
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

# 1. 경로 설정
MODEL_PATH = r"E:\AIVLE_10team\results\bestYOLOm5080model.pt"
VIDEO_PATH = r"E:\test\cctv_EXCO_test_output.mp4"

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

print("🎬 동영상 YOLO 추론 시작! (종료하려면 영상 창에서 'q' 키를 누르세요)\n")

confidences = [] # 신뢰도 점수 모음
total_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # 영상 끝
        
    total_frames += 1
    
    # YOLO 추론 실행 (ByteTrack 적용)
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            # 신뢰도 점수(Confidence) 추출 (예: 0.85 = 85% 확신)
            confs = boxes.conf.cpu().numpy()
            confidences.extend(confs)
            
            # 화면에 박스 및 ID 시각화해 주는 이미지를 가져옴
            annotated_frame = r.plot()
        else:
            annotated_frame = frame

    # 실시간 화면 출력
    cv2.imshow("YOLO11s + ByteTrack Test", annotated_frame)
    
    # 'q' 키를 누르면 중도 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 임시 입력 파일 정리
if temp_read_path and os.path.exists(temp_read_path):
    try:
        os.remove(temp_read_path)
    except Exception:
        pass

# ==========================================
# 📊 간단한 성능(신뢰도) 결과 출력
# ==========================================
print("\n" + "="*40)
print("📊 YOLO 동영상 테스트 결과 리포트")
print("="*40)
print(f"🎥 총 테스트 프레임 수: {total_frames} 프레임")

if confidences:
    avg_conf = np.mean(confidences) * 100
    print(f"👥 탐지된 총 사람 수 (누적): {len(confidences)}명")
    print(f"🎯 평균 인식 신뢰도(Confidence): {avg_conf:.2f}%")
else:
    print("❌ 사람을 한 명도 찾지 못했습니다.")
print("="*40)