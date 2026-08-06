# 03_postprocess.py
"""Post‑processing utilities for CSRNet output.
- Convert density map to pedestrian count per frame
- Generate BEV coordinates and risk score
- Prepare payload for Supabase insertion
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def density_to_count(density_map: np.ndarray) -> int:
    """Sum of the density map approximates number of pedestrians.
    Returns an integer count.
    """
    count = density_map.sum()
    return int(round(count))

def generate_bev_points(density_map: np.ndarray, scale: float = 0.1) -> list[dict]:
    """Create a list of BEV point dictionaries from the density map.
    Each point contains approximate (x, y) in world coordinates and a weight.
    """
    points = []
    h, w = density_map.shape
    for y in range(h):
        for x in range(w):
            weight = float(density_map[y, x])
            if weight < 1e-3:
                continue
            # simple linear scaling to world coordinates (example)
            world_x = x * scale
            world_y = y * scale
            points.append({"x": world_x, "y": world_y, "weight": weight})
    logger.debug("Generated %d BEV points", len(points))
    return points

def prepare_db_record(frame_id: int, timestamp: str, density_map: np.ndarray) -> dict:
    """Assemble a single database record for Supabase.
    - `frame_id`: sequential integer
    - `timestamp`: ISO8601 string
    - `ped_count`: integer count from density map
    - `bev_points`: list of point dicts (jsonb column)
    """
    count = density_to_count(density_map)
    bev = generate_bev_points(density_map)
    record = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "ped_count": count,
        "bev_points": bev,
    }
    return record
