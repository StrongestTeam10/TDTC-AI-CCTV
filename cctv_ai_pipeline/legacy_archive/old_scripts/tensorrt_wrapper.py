# tensorrt_wrapper.py
# PyTorch Native FP16 & TensorRT Engine Wrapper for CSRNet

import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import models

# 석훈님 CSRNet 모델 정의
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

class SharedCSRNetEngine:
    """
    단일 모델 인스턴스를 공유하며 batch_size=3 (또는 N<=3) Batch 추론을 수행하는 전담 엔진.
    PyTorch Native FP16/Half precision 기반으로 동작하며, TensorRT .engine 파일 존재 시 연동 가능.
    """
    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.trt_engine = None

        print(f"[ENGINE INIT] CSRNet Shared Model 로드 중... (Device: {self.device}, FP16: {self.fp16})")
        self.model = CSRNet().to(self.device)

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            if self.fp16:
                self.model = self.model.half()
            print(f"[ENGINE SUCCESS] CSRNet 모델 가중치 로드 완료: {os.path.basename(model_path)}")
        else:
            raise FileNotFoundError(f"CSRNet 가중치 파일을 찾을 수 없습니다: {model_path}")

        # ONNX / TensorRT 엔진 파이프라인 검사 (지원 시)
        engine_file = model_path.replace(".pth", ".engine")
        if os.path.exists(engine_file):
            print(f"[TRT FOUND] TensorRT 엔진 발견: {engine_file}")
            # TRT 래퍼 연동 로직 (선택적)

    @torch.no_grad()
    def infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        batch_tensor: (N, 3, H, W) 형태의 PyTorch 텐서 (CPU 또는 GPU)
        반환: (N, 1, H_out, W_out) 밀도 지도 텐서
        """
        if batch_tensor.device != self.device:
            batch_tensor = batch_tensor.to(self.device)

        if self.fp16 and batch_tensor.dtype != torch.float16:
            batch_tensor = batch_tensor.half()

        # PyTorch Native Batch 추론
        output = self.model(batch_tensor)
        return output

def export_to_onnx(model_path: str, onnx_output_path: str, batch_size: int = 3):
    """CSRNet 모델을 ONNX 형식으로 Export 유틸리티"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 720, 1280, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[ONNX SUCCESS] ONNX 모델 Export 완료: {onnx_output_path}")
