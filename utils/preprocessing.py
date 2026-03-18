"""
Space Debris AI — Data Preprocessing & Augmentation Utilities
Provides augmentation transforms for training data robustness.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def random_rotation(image: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    """Apply random rotation to image."""
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(image: np.ndarray, factor_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    """Randomly adjust image brightness."""
    factor = np.random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, factor_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    """Randomly adjust image contrast."""
    factor = np.random.uniform(*factor_range)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 15.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def horizontal_flip(image: np.ndarray, probability: float = 0.5) -> np.ndarray:
    """Randomly flip the image horizontally."""
    if np.random.random() < probability:
        return cv2.flip(image, 1)
    return image


def apply_augmentations(
    image: np.ndarray,
    rotate: bool = True,
    brightness: bool = True,
    contrast: bool = True,
    noise: bool = True,
    flip: bool = True,
) -> np.ndarray:
    """Apply a pipeline of random augmentations."""
    augmented = image.copy()

    if rotate:
        augmented = random_rotation(augmented)
    if brightness:
        augmented = adjust_brightness(augmented)
    if contrast:
        augmented = adjust_contrast(augmented)
    if noise:
        augmented = add_gaussian_noise(augmented)
    if flip:
        augmented = horizontal_flip(augmented)

    return augmented


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess an image for model inference."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    if normalize:
        resized = resized.astype(np.float32) / 255.0
    return resized


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk."""
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Could not load image from {path}")
    return image
