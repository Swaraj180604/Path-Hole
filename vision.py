"""Optional camera-based detection with YOLOv8 (needs `ultralytics`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["class", "confidence", "x1", "y1", "x2", "y2"]


def is_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def predict(model, rgb: np.ndarray, conf: float = 0.4):
    """Run detection on an RGB image. Returns (annotated RGB image, detections df)."""
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    annotated = np.ascontiguousarray(res.plot()[..., ::-1])
    rows = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "class": res.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return annotated, pd.DataFrame(rows, columns=COLUMNS)
