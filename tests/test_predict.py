"""
Tests for inference/predict.py
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict import generate_demo_detections, compute_statistics, detect_debris


class TestGenerateDemoDetections:
    """Tests for the synthetic demo detection generator."""

    def test_returns_list(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image)
        assert isinstance(result, list)

    def test_returns_correct_count(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=5)
        assert len(result) == 5

    def test_detection_has_required_keys(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=1)
        det = result[0]
        assert "bbox" in det
        assert "confidence" in det
        assert "label" in det
        assert "class_id" in det

    def test_bbox_format(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=1)
        bbox = result[0]["bbox"]
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert x1 < x2, "x1 should be less than x2"
        assert y1 < y2, "y1 should be less than y2"

    def test_confidence_range(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=10)
        for det in result:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_sorted_by_confidence_descending(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=10)
        confs = [d["confidence"] for d in result]
        assert confs == sorted(confs, reverse=True)

    def test_bbox_within_image_bounds(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_demo_detections(image, num_detections=20)
        for det in result:
            x1, y1, x2, y2 = det["bbox"]
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 640 and y2 <= 480

    def test_default_count_range(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            result = generate_demo_detections(image)
            assert 3 <= len(result) <= 8


class TestComputeStatistics:
    """Tests for statistics computation."""

    def test_empty_detections(self):
        stats = compute_statistics([])
        assert stats["total_debris"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["max_confidence"] == 0.0
        assert stats["risk_level"] == "Low"

    def test_single_detection(self):
        detections = [{"confidence": 0.85, "bbox": (10, 10, 50, 50), "label": "debris"}]
        stats = compute_statistics(detections)
        assert stats["total_debris"] == 1
        assert stats["avg_confidence"] == 0.85
        assert stats["max_confidence"] == 0.85
        assert stats["risk_level"] == "Low"

    def test_high_risk_level(self):
        detections = [{"confidence": 0.9} for _ in range(7)]
        stats = compute_statistics(detections)
        assert stats["risk_level"] == "High"

    def test_medium_risk_level(self):
        detections = [{"confidence": 0.8} for _ in range(4)]
        stats = compute_statistics(detections)
        assert stats["risk_level"] == "Medium"

    def test_low_risk_level(self):
        detections = [{"confidence": 0.7} for _ in range(2)]
        stats = compute_statistics(detections)
        assert stats["risk_level"] == "Low"

    def test_avg_confidence_calculation(self):
        detections = [
            {"confidence": 0.6},
            {"confidence": 0.8},
            {"confidence": 1.0},
        ]
        stats = compute_statistics(detections)
        assert stats["avg_confidence"] == 0.8

    def test_max_and_min_confidence(self):
        detections = [
            {"confidence": 0.3},
            {"confidence": 0.9},
            {"confidence": 0.6},
        ]
        stats = compute_statistics(detections)
        assert stats["max_confidence"] == 0.9
        assert stats["min_confidence"] == 0.3


class TestDetectDebris:
    """Tests for the main detection entry point."""

    def test_demo_mode_returns_detections(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detect_debris(image, use_demo=True)
        assert len(result) > 0

    def test_missing_model_falls_back_to_demo(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detect_debris(image, model_path="nonexistent_model.pt")
        assert len(result) > 0

    def test_none_model_path_uses_demo(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detect_debris(image, model_path=None)
        assert len(result) > 0
