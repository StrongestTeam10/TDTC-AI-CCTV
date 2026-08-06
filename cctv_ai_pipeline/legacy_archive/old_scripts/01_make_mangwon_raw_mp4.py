import os
import glob
import cv2
import numpy as np
import re

MANGWON_RAW_DIR = r"E:\test\cctv_망원시장"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 루트 results 폴더로 단일 통일
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))

OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "cctv_mangwon_raw_video.mp4")

def cv2_imread_korean(file_path):
    """한글 경로 파일명을 지원하는 OpenCV imread 헬퍼"""
    try:
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[READ ERROR] {file_path}: {e}")
        return None

def get_frame_num(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0

raw_files = sorted(glob.glob(os.path.join(MANGWON_RAW_DIR, "*.jpg")), key=get_frame_num)

if not raw_files:
    print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {MANGWON_RAW_DIR}")
    exit(1)

print(f"[LOAD] E:\\test\\cctv_망원시장 프레임 수: {len(raw_files)}개")

first_img = cv2_imread_korean(raw_files[0])
if first_img is None:
    print(f"[ERROR] 이미지를 읽을 수 없습니다: {raw_files[0]}")
    exit(1)

img_h, img_w, _ = first_img.shape
print(f"[INFO] 원본 이미지 해상도: {img_w}x{img_h}")

fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = 10.0
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (img_w, img_h))

if not video_writer.isOpened():
    print(f"[ERROR] VideoWriter 생성 실패: {OUTPUT_VIDEO_PATH}")
    exit(1)

print("[PROCESSING] E:\\test\\cctv_망원시장 원본 이미지를 H.264 MP4로 인코딩 중...")
count = 0
for img_path in raw_files:
    img = cv2_imread_korean(img_path)
    if img is not None:
        video_writer.write(img)
        count += 1

video_writer.release()
print(f"[SUCCESS] 원본 MP4 생성 완료 ({count} frames): {OUTPUT_VIDEO_PATH}")
