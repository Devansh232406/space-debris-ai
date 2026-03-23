"""
Tests for utils/preprocessing.py
"""

import numpy as np
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (
    random_rotation,
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    horizontal_flip,
    apply_augmentations,
    preprocess_image,
)


def make_test_image(h=100, w=150):
    """Create a simple test image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestRandomRotation:
    def test_preserves_shape(self):
        img = make_test_image()
        result = random_rotation(img)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        img = make_test_image()
        result = random_rotation(img)
        assert result.dtype == np.uint8


class TestAdjustBrightness:
    def test_preserves_shape(self):
        img = make_test_image()
        result = adjust_brightness(img)
        assert result.shape == img.shape

    def test_values_in_valid_range(self):
        img = make_test_image()
        result = adjust_brightness(img)
        assert result.min() >= 0 and result.max() <= 255


class TestAdjustContrast:
    def test_preserves_shape(self):
        img = make_test_image()
        result = adjust_contrast(img)
        assert result.shape == img.shape

    def test_values_in_valid_range(self):
        img = make_test_image()
        result = adjust_contrast(img)
        assert result.min() >= 0 and result.max() <= 255


class TestAddGaussianNoise:
    def test_preserves_shape(self):
        img = make_test_image()
        result = add_gaussian_noise(img)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        img = make_test_image()
        result = add_gaussian_noise(img)
        assert result.dtype == np.uint8

    def test_values_in_valid_range(self):
        img = make_test_image()
        result = add_gaussian_noise(img)
        assert result.min() >= 0 and result.max() <= 255


class TestHorizontalFlip:
    def test_preserves_shape(self):
        img = make_test_image()
        result = horizontal_flip(img, probability=1.0)
        assert result.shape == img.shape

    def test_flips_when_probability_one(self):
        img = make_test_image()
        result = horizontal_flip(img, probability=1.0)
        # First column of original should be last column of flipped
        np.testing.assert_array_equal(result[:, 0, :], img[:, -1, :])

    def test_no_flip_when_probability_zero(self):
        img = make_test_image()
        result = horizontal_flip(img, probability=0.0)
        np.testing.assert_array_equal(result, img)


class TestApplyAugmentations:
    def test_preserves_shape(self):
        img = make_test_image()
        result = apply_augmentations(img)
        assert result.shape == img.shape

    def test_does_not_mutate_input(self):
        img = make_test_image()
        original = img.copy()
        apply_augmentations(img)
        np.testing.assert_array_equal(img, original)

    def test_no_augmentations(self):
        img = make_test_image()
        result = apply_augmentations(
            img, rotate=False, brightness=False, contrast=False, noise=False, flip=False
        )
        np.testing.assert_array_equal(result, img)


class TestPreprocessImage:
    def test_resizes_to_target(self):
        img = make_test_image(200, 300)
        result = preprocess_image(img, target_size=(640, 640))
        assert result.shape[:2] == (640, 640)

    def test_normalization(self):
        img = make_test_image()
        result = preprocess_image(img, normalize=True)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_no_normalization(self):
        img = make_test_image()
        result = preprocess_image(img, target_size=(100, 150), normalize=False)
        assert result.dtype == np.uint8
