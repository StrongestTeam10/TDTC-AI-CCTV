"""
server/models.py - CSRNet 모델, 좌표 변환, 피크 추출 및 모자이크(가우시안 블러) 비식별화 헬퍼
"""

import os
import json
import cv2
import numpy as np
import scipy.ndimage as ndimage
from server.config import BASE_DIR, MODELS_DIR

torch_available = False
CSRNet = None
csr_transform = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    torch_available = True

    class CSRNetClass(nn.Module):
        """군중 밀도 추정 CSRNet 모델 (VGG16 백본 기반)"""
        def __init__(self):
            super(CSRNetClass, self).__init__()
            vgg = models.vgg16(weights=None)
            features = list(vgg.features.children())
            self.frontend = nn.Sequential(*features[0:23])
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # backend.0
                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # backend.2
                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # backend.4
                nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # backend.6
                nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),  # backend.8
                nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),   # backend.10
                nn.Conv2d(64, 1, 1)                                                      # backend.12
            )

        def forward(self, x):
            return self.backend(self.frontend(x))

    CSRNet = CSRNetClass

    csr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

except Exception as e:
    print(f"[AI Server Models Setup Warning] PyTorch/torchvision 로드 불가: {e}")

# =========================================================================
# 호모그래피 변환 행렬 및 ROI 다각형 로드
# =========================================================================
H_MATRIX = None
ROI_2D_POLY = None
try:
    config_path = os.path.join(BASE_DIR, "cctv_upload", "core", "zones_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, "core", "zones_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            zones_config = json.load(f)
            zone_meta = zones_config.get("1", {})  # 기본 Zone 1
            H_MATRIX = np.array(zone_meta.get("h_matrix"))
            ROI_2D_POLY = np.array(zone_meta.get("roi_polygon"), dtype=np.int32)
            print(f"[AI Server Models] zones_config.json에서 기본 캘리브레이션 및 ROI 로드 완료!")
except Exception as e:
    print(f"[AI Server Models Warning] zones_config.json 로드 실패, 기본값 사용: {e}")

if H_MATRIX is None:
    H_MATRIX = np.array([
        [ 0.015, -0.002, -8.50],
        [ 0.001,  0.022, -2.10],
        [ 0.000,  0.001,  1.00]
    ])
if ROI_2D_POLY is None:
    ROI_2D_POLY = np.array([
        [70, 1000],
        [320, 220],
        [940, 220],
        [1120, 900]
    ], dtype=np.int32)


def transform_pixel_to_bev(u: float, v: float, H: np.ndarray):
    """2D 픽셀 좌표 (u, v) → BEV 물리 좌표 (X, Y) 변환"""
    pt = np.array([u, v, 1.0]).reshape(3, 1)
    bev_pt = np.dot(H, pt)
    bev_pt /= bev_pt[2]
    return float(bev_pt[0][0]), float(bev_pt[1][0])


def extract_peaks_from_density(density_map: np.ndarray, threshold: float = 0.005):
    """CSRNet 밀도 맵에서 보행자 중심점 Peak (u, v) 좌표 추출"""
    neighborhood_size = 5
    data_max = ndimage.maximum_filter(density_map, neighborhood_size)
    maxima = (density_map == data_max)
    data_min = ndimage.minimum_filter(density_map, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[~diff] = 0

    labeled, _ = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    peaks = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2.0
        y_center = (dy.start + dy.stop - 1) / 2.0
        peaks.append((x_center, y_center))
    return peaks


def apply_mosaic(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, neighbor: int = 15):
    """지정된 바운딩 박스/머리 영역에 부드러운 타원형 가우시안 블러 비식별화(모자이크) 적용"""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return img

    roi = img[y1:y2, x1:x2]
    kernel_w = max(5, (w // 2) * 2 + 1)
    kernel_h = max(5, (h // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (kernel_w, kernel_h), sigmaX=20, sigmaY=20)

    # 타원형 페더링 마스크를 생성하여 경계면을 자연스럽게 알파 블렌딩
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1.0, -1)
    if min(w, h) >= 15:
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.expand_dims(mask, axis=2)

    blended = (blurred.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    img[y1:y2, x1:x2] = blended
    return img
