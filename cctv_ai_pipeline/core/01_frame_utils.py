# 01_frame_utils.py
"""Core utilities for frame handling.
- Korean‑path image read helper
- RTSP frame capture helper
- Frame ordering utilities
"""

import os
import cv2
import numpy as np
import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def cv2_imread_korean(file_path: str) -> np.ndarray | None:
    """Read an image with a Korean path safely using numpy + cv2.
    Returns None on error.
    """
    try:
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"[READ ERROR] {file_path}: {e}")
        return None

def get_frame_num(filename: str) -> int:
    """Extract the numeric part of a filename like '123.jpg' for sorting.
    Returns 0 if no match.
    """
    match = re.search(r"(\d+)\.jpg$", filename)
    return int(match.group(1)) if match else 0

def collect_image_files(dir_path: str) -> List[str]:
    """Collect all *.jpg files from *dir_path* sorted by frame number.
    """
    import glob
    raw_files = sorted(
        glob.glob(os.path.join(dir_path, "*.jpg")),
        key=get_frame_num,
    )
    if not raw_files:
        logger.error(f"[ERROR] No image files found in {dir_path}")
    return raw_files

def read_rtsp_stream(url: str, timeout: int = 5) -> cv2.VideoCapture:
    """Create a cv2.VideoCapture for an RTSP stream.
    Raises RuntimeError if the stream cannot be opened.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP stream: {url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

def grab_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray | None]:
    """Grab a single frame from an opened VideoCapture.
    Returns (ret, frame).
    """
    ret, frame = cap.read()
    return ret, frame if ret else (False, None)
