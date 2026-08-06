# make_mp4.py: 특정 CCTV 이미지 폴더 내의 연속된 .jpg 프레임 이미지들을 하나의 .mp4 동영상 파일로 결합하여 저장하는 스크립트입니다.
import os
# pyrefly: ignore [missing-import]
import cv2
import glob
import numpy as np
import shutil

# 💡 한글 경로 지원용 이미지 로드 함수 (예외 처리 강화)
def imread_korean(path):
    if not os.path.exists(path):
        return None
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 로드 실패 ({path}): {e}")
        return None

# 💡 한글 경로 지원용 VideoWriter 헬퍼
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

# 1. 이미지 폴더 및 저장할 mp4 파일 경로
IMAGE_DIR = os.environ.get("IMAGE_DIR", r"E:\test\cctv_cafe")
OUTPUT_MP4 = os.environ.get("OUTPUT_MP4", r"E:\test\cctv_cafe_output.mp4")
FPS = int(os.environ.get("FPS", "15"))  # 초당 프레임 수 (10~30 사이)

# 2. 이미지 파일 목록 가져오기 및 정렬
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

MAX_FRAMES = os.environ.get("MAX_FRAMES")
if MAX_FRAMES is not None and MAX_FRAMES.isdigit():
    max_f = int(MAX_FRAMES)
    image_paths = image_paths[:max_f]
    print(f"[INFO] 고속 데모 모드: 프레임 수를 {max_f}개로 제한하여 처리합니다.")

if not image_paths:
    print(f"❌ '{IMAGE_DIR}' 폴더에 jpg 파일이 없습니다!")
    exit()

# 3. 첫 번째 이미지에서 영상 해상도(가로, 세로) 자동 추출 (한글 경로 대응)
first_img = imread_korean(image_paths[0])

if first_img is None:
    print(f"❌ 첫 번째 이미지를 읽을 수 없습니다: {image_paths[0]}")
    exit()

height, width, _ = first_img.shape

# 4. mp4 비디오 라이터(VideoWriter) 설정 (한글 경로 대응 적용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer, temp_write_path = get_safe_video_writer(OUTPUT_MP4, fourcc, FPS, (width, height))

if video_writer is None or not video_writer.isOpened():
    print(f"❌ 비디오 저장 라이터를 열 수 없습니다. 경로를 확인해 주세요: {OUTPUT_MP4}")
    exit()

print(f"🎬 총 {len(image_paths)}장의 이미지를 mp4 동영상으로 변환하는 중...")

# 5. 이미지를 하나씩 읽어서 영상에 이어 붙이기
for idx, img_path in enumerate(image_paths):
    img = imread_korean(img_path)
    if img is not None:
        video_writer.write(img)
    
    if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
        print(f"  └ 진행률: {idx + 1}/{len(image_paths)} 완료")

# 6. 저장 완료 및 리소스 해제
video_writer.release()

# 임시 파일 경로를 원래 최종 목적지 경로로 이동
if temp_write_path and os.path.exists(temp_write_path):
    if os.path.exists(OUTPUT_MP4):
        os.remove(OUTPUT_MP4)
    shutil.move(temp_write_path, OUTPUT_MP4)

print(f"\n✅ 동영상 변환 완벽 성공!")
print(f"📁 저장된 파일: {OUTPUT_MP4}")