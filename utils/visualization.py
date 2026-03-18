"""
Space Debris AI — Detection Visualization Utilities
Draw bounding boxes, labels, and confidence scores on images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# Risk-level color scheme (BGR for OpenCV)
RISK_COLORS = {
    "High":   (0, 0, 255),     # Red
    "Medium": (0, 200, 255),   # Yellow-Orange
    "Low":    (0, 220, 0),     # Green
}

BOX_COLOR = (0, 255, 128)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)


def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.

    Each detection dict should have:
        - bbox: (x1, y1, x2, y2) in pixel coords
        - confidence: float 0-1
        - label: str
    """
    annotated = image.copy()

    for det in detections:
        bbox = det["bbox"]
        conf = det["confidence"]
        label = det.get("label", "debris")
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Determine color based on confidence
        if conf >= 0.8:
            color = RISK_COLORS["High"]
        elif conf >= 0.5:
            color = RISK_COLORS["Medium"]
        else:
            color = RISK_COLORS["Low"]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label text
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Background for text
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return annotated


def create_stats_overlay(
    image: np.ndarray,
    total_debris: int,
    avg_confidence: float,
    risk_level: str,
) -> np.ndarray:
    """Add a semi-transparent statistics overlay to the image."""
    overlay = image.copy()
    h, w = image.shape[:2]

    # Semi-transparent panel
    panel_h = 100
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    y_start = h - panel_h + 25
    stats = [
        f"Total Debris: {total_debris}",
        f"Avg Confidence: {avg_confidence:.2f}",
        f"Risk Level: {risk_level}",
    ]
    for i, text in enumerate(stats):
        cv2.putText(
            annotated, text,
            (15, y_start + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return annotated


def get_risk_level(debris_count: int) -> str:
    """Classify risk level based on debris count."""
    if debris_count >= 6:
        return "High"
    elif debris_count >= 3:
        return "Medium"
    else:
        return "Low"


def get_risk_color_hex(risk_level: str) -> str:
    """Get hex color for a risk level (for web/dashboard use)."""
    mapping = {
        "High": "#FF4444",
        "Medium": "#FFAA00",
        "Low": "#44DD44",
    }
    return mapping.get(risk_level, "#FFFFFF")
