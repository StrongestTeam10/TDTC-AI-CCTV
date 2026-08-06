import os
import glob
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 루트 results 폴더로 단일 통일
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
FRAMES_DIR = os.path.join(RESULTS_DIR, "frames")

OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "cctv_raw_video.mp4")

frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
if not frame_files:
    print(f"[ERROR] No jpg frame images found in: {FRAMES_DIR}")
    exit(1)

first_img = cv2.imread(frame_files[0])
if first_img is None:
    print(f"[ERROR] Cannot read frame image: {frame_files[0]}")
    exit(1)

img_h, img_w, _ = first_img.shape
print(f"[INFO] Raw CCTV Resolution: {img_w}x{img_h}, Frames: {len(frame_files)}")

fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = 10.0
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (img_w, img_h))

if not video_writer.isOpened():
    print(f"[ERROR] Cannot open VideoWriter for {OUTPUT_VIDEO_PATH}")
    exit(1)

for idx, img_path in enumerate(frame_files):
    frame_img = cv2.imread(img_path)
    if frame_img is None:
        continue
    video_writer.write(frame_img)

video_writer.release()
print(f"[SUCCESS] Clean Raw CCTV Video created: {OUTPUT_VIDEO_PATH}")
