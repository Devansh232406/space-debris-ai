"""
Space Debris AI — Inference Pipeline
Runs YOLOv8 detection on uploaded images.
Falls back to synthetic demo detections when no custom model is available.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def generate_demo_detections(image: np.ndarray, num_detections: int = None) -> List[Dict]:
    """
    Generate realistic synthetic debris detections for demo purposes.
    Used when no trained model is available.
    """
    h, w = image.shape[:2]
    if num_detections is None:
        num_detections = np.random.randint(3, 9)

    detections = []
    for i in range(num_detections):
        # Random bounding box
        box_w = np.random.randint(int(w * 0.03), int(w * 0.12))
        box_h = np.random.randint(int(h * 0.03), int(h * 0.12))
        x1 = np.random.randint(10, max(11, w - box_w - 10))
        y1 = np.random.randint(10, max(11, h - box_h - 10))
        x2 = x1 + box_w
        y2 = y1 + box_h

        confidence = round(np.random.uniform(0.55, 0.98), 2)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "label": "space_debris",
            "class_id": 0,
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def run_yolo_inference(image: np.ndarray, model_path: str, conf_threshold: float = 0.25) -> List[Dict]:
    """
    Run YOLOv8 inference on an image.

    Args:
        image: BGR numpy array
        model_path: Path to YOLO .pt weights
        conf_threshold: Minimum confidence threshold

    Returns:
        List of detection dicts with bbox, confidence, label, class_id
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Warning: ultralytics not installed, using demo detections")
        return generate_demo_detections(image)

    try:
        model = YOLO(model_path)
        results = model(image, conf=conf_threshold, verbose=False)
    except Exception as e:
        print(f"Warning: Model inference failed ({e}), using demo detections")
        return generate_demo_detections(image)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = result.names.get(cls_id, "space_debris")

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": round(conf, 2),
                "label": label,
                "class_id": cls_id,
            })

    return detections


def detect_debris(
    image: np.ndarray,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    use_demo: bool = False,
) -> List[Dict]:
    """
    Main detection entry point.

    Attempts to use a trained YOLO model. Falls back to demo detections
    if no model is found or inference fails.
    """
    if use_demo or model_path is None:
        return generate_demo_detections(image)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Model not found at {model_path}, using demo detections")
        return generate_demo_detections(image)

    return run_yolo_inference(image, str(model_file), conf_threshold)


def compute_statistics(detections: List[Dict]) -> Dict:
    """Compute summary statistics from detections."""
    if not detections:
        return {
            "total_debris": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "risk_level": "Low",
        }

    confidences = [d["confidence"] for d in detections]
    total = len(detections)

    if total >= 6:
        risk = "High"
    elif total >= 3:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "total_debris": total,
        "avg_confidence": round(np.mean(confidences), 2),
        "max_confidence": round(max(confidences), 2),
        "min_confidence": round(min(confidences), 2),
        "risk_level": risk,
    }
