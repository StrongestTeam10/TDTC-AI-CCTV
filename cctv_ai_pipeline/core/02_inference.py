# 02_inference.py
"""Core inference utilities.
- Load CSRNet model (optional TensorRT/torch.compile)
- Provide a single `run_inference(frame)` entry point returning raw model output.
"""

import torch
import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Adjust these paths as needed
MODEL_CHECKPOINT = Path(__file__).parent.parent / "models" / "csrnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() -> torch.nn.Module:
    """Load CSRNet model.
    Supports optional TorchScript or TensorRT compiled version if available.
    """
    try:
        # Placeholder: you should replace with actual CSRNet definition
        # pyrefly: ignore [missing-import]
        from .model_def import CSRNet  # expect a model_def.py in core
        model = CSRNet()
        state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        logger.info("CSRNet model loaded on %s", DEVICE)
        return model
    except Exception as e:
        logger.error("Failed to load CSRNet model: %s", e)
        raise

# Load once at import time for efficiency
_MODEL = load_model()

def run_inference(frame: "np.ndarray") -> torch.Tensor:
    """Run CSRNet inference on a single frame.
    Input: BGR numpy array (as read by OpenCV)
    Output: Tensor of shape (1, 1, H, W) – density map.
    """
    import numpy as np
    # Convert BGR -> RGB, normalize, to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _MODEL(tensor)
    return out.squeeze(0).cpu()
