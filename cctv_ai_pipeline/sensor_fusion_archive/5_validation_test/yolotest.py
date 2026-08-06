# yolotest.py: YOLOv8 모델로 CCTV 테스트 이미지 폴더의 인원을 검출하고 발밑 픽셀 좌표(x, y)를 터미널에 출력하는 간단한 테스트 스크립트입니다.
import os
import json
import glob
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ==========================================
# 1. 경로 설정 (태훈이가 알려준 절대경로)
# ==========================================
MODEL_PATH = r"E:\AIVLE_10team\results\bestYOLOm5080model.pt"
IMAGE_DIR  = r"E:\test\cctv_EXCO_test"
LABEL_DIR  = r"E:\test\cctv_EXCO_test_label"

# ==========================================
# 2. YOLO 모델 로드
# ==========================================
print("📂 YOLO 모델을 로드하는 중...")
model = YOLO(MODEL_PATH)
print("✅ 모델 로드 완료!\n")

# ==========================================
# 3. 이미지 파일 목록 가져오기 (.jpg)
# ==========================================
# E:\test\cctv_EXCO_test 안의 모든 jpg 파일 검색
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

if not image_paths:
    print(f"❌ 경고: '{IMAGE_DIR}' 폴더에 .jpg 파일이 없습니다. 경로를 확인해 주세요!")
else:
    print(f"🖼️ 총 {len(image_paths)}개의 JPG 이미지를 찾았습니다. 추론을 시작합니다.\n")

# ==========================================
# 4. 이미지별 추론 및 (x, y) 좌표 추출
# ==========================================
for img_path in image_paths:
    file_name = os.path.basename(img_path) # 예: image_001.jpg
    base_name = os.path.splitext(file_name)[0] # 예: image_001
    
    # ------------------------------------------
    # A. 석훈님 YOLO 모델로 이미지 추론 (2D 좌표)
    # ------------------------------------------
    results = model.predict(source=img_path, verbose=False)
    
    yolo_foot_points = []
    
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # 🔥 핵심: 사람 발밑 중앙 픽셀 좌표 (x, y)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                yolo_foot_points.append((foot_x, foot_y))
                
    print(f"[{file_name}] YOLO가 찾은 사람 수: {len(yolo_foot_points)}명")
    print(f"  └ 발밑 픽셀 좌표(x, y): {yolo_foot_points}")
    
    # ------------------------------------------
    # B. (선택사항) AI허브 정답 라벨(JSON) 같이 확인하기
    # ------------------------------------------
    json_path = os.path.join(LABEL_DIR, f"{base_name}.json")
    if os.path.exists(json_path):
# encoding을 'utf-8-sig'로 변경!
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            label_data = json.load(f)
            # print(f"  └ 정답 JSON 파일 읽기 성공: {json_path}")
            
    # ========================================================
    # 💡 [태훈이의 AI 연동 파트]
    # 여기서 뽑힌 `yolo_foot_points` [ (x1, y1), (x2, y2), ... ] 를
    # 태훈이의 BEV(호모그래피) 변환 함수 -> DBSCAN 함수에 넘겨주면 끝!
    #
    # 예시:
    # bev_points = convert_to_bev(yolo_foot_points)
    # clusters = run_dbscan(bev_points)
    # ========================================================
    print("-" * 50)