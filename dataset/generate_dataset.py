"""
Space Debris AI — Synthetic Dataset Generator
Generates labeled satellite-like images with procedural space debris objects.
Outputs YOLO-format annotations for training YOLOv8.

Usage:
    python dataset/generate_dataset.py
    python dataset/generate_dataset.py --train 300 --valid 75 --test 75
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path


# ─── Configuration ───────────────────────────────────────
IMG_SIZE = 640
NUM_CLASSES = 1  # 0: space_debris


def create_starfield_background(size: int = IMG_SIZE) -> np.ndarray:
    """Generate a realistic dark space background with stars."""
    # Dark gradient background
    bg = np.zeros((size, size, 3), dtype=np.uint8)

    # Add subtle nebula-like color variation
    for _ in range(3):
        cx, cy = np.random.randint(0, size, 2)
        radius = np.random.randint(100, 300)
        color_shift = np.random.randint(0, 15, 3)
        y_grid, x_grid = np.ogrid[:size, :size]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        mask = np.clip(1 - dist / radius, 0, 1)
        for c in range(3):
            bg[:, :, c] = np.clip(
                bg[:, :, c].astype(np.float32) + mask * color_shift[c], 0, 255
            ).astype(np.uint8)

    # Add stars (small bright dots)
    num_stars = np.random.randint(100, 400)
    for _ in range(num_stars):
        x, y = np.random.randint(0, size, 2)
        brightness = np.random.randint(80, 255)
        star_size = np.random.choice([1, 1, 1, 2])  # mostly single pixels
        color = (brightness, brightness, min(255, brightness + np.random.randint(0, 30)))
        cv2.circle(bg, (x, y), star_size, color, -1)

    # Slight Gaussian blur for realism
    if np.random.random() < 0.3:
        bg = cv2.GaussianBlur(bg, (3, 3), 0)

    return bg


def draw_debris_fragment(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Draw an irregular debris fragment."""
    num_points = np.random.randint(4, 8)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
    radii = np.random.uniform(size * 0.4, size * 0.9, num_points)

    points = []
    for angle, r in zip(angles, radii):
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        points.append([px, py])

    points = np.array(points, dtype=np.int32)

    # Metallic gray/silver colors
    base_brightness = np.random.randint(120, 230)
    color = (
        base_brightness + np.random.randint(-20, 20),
        base_brightness + np.random.randint(-20, 20),
        base_brightness + np.random.randint(-10, 30),
    )
    color = tuple(int(np.clip(c, 0, 255)) for c in color)

    cv2.fillPoly(image, [points], color)

    # Add highlight edge
    if np.random.random() < 0.5:
        highlight = tuple(min(255, c + 40) for c in color)
        cv2.polylines(image, [points], True, highlight, 1)

    return image


def draw_debris_sphere(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Draw a round debris object (defunct satellite, bolt, etc)."""
    radius = max(2, size // 2)
    base_brightness = np.random.randint(100, 220)
    color = (base_brightness, base_brightness + 10, base_brightness + 20)

    cv2.circle(image, (cx, cy), radius, color, -1)

    # Highlight (sunlit side)
    highlight_offset = np.random.randint(-radius // 3, radius // 3 + 1)
    highlight_color = tuple(min(255, c + 50) for c in color)
    cv2.circle(
        image, (cx + highlight_offset, cy - abs(highlight_offset)),
        max(1, radius // 3), highlight_color, -1,
    )

    return image


def draw_debris_streak(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Draw a motion-streaked debris (fast-moving object)."""
    angle = np.random.uniform(0, np.pi)
    length = size * np.random.uniform(1.5, 3.0)
    thickness = max(1, int(size * 0.3))

    x1 = int(cx - length / 2 * np.cos(angle))
    y1 = int(cy - length / 2 * np.sin(angle))
    x2 = int(cx + length / 2 * np.cos(angle))
    y2 = int(cy + length / 2 * np.sin(angle))

    brightness = np.random.randint(140, 250)
    color = (brightness, brightness, min(255, brightness + 20))

    cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Bright center
    cv2.circle(image, (cx, cy), max(1, thickness), (255, 255, 255), -1)

    return image


def generate_single_image(img_size: int = IMG_SIZE, min_debris: int = 1, max_debris: int = 10):
    """
    Generate one labeled image with debris objects.

    Returns:
        image: np.ndarray (H, W, 3) BGR
        annotations: list of (class_id, x_center, y_center, width, height) normalized
    """
    image = create_starfield_background(img_size)
    num_debris = np.random.randint(min_debris, max_debris + 1)
    annotations = []

    for _ in range(num_debris):
        # Random size (in pixels)
        debris_size = np.random.randint(6, 45)

        # Random position (keep within bounds)
        margin = debris_size + 5
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        # Choose debris style
        style = np.random.choice(["fragment", "sphere", "streak"], p=[0.5, 0.3, 0.2])

        if style == "fragment":
            image = draw_debris_fragment(image, cx, cy, debris_size)
            bbox_size = int(debris_size * 1.8)
        elif style == "sphere":
            image = draw_debris_sphere(image, cx, cy, debris_size)
            bbox_size = int(debris_size * 1.2)
        else:
            image = draw_debris_streak(image, cx, cy, debris_size)
            bbox_size = int(debris_size * 3.0)

        # YOLO format: class_id x_center y_center width height (all normalized 0-1)
        x_center = cx / img_size
        y_center = cy / img_size
        width = min(bbox_size / img_size, 1.0)
        height = min(bbox_size / img_size, 1.0)

        # Clamp to valid range
        x_center = np.clip(x_center, width / 2, 1 - width / 2)
        y_center = np.clip(y_center, height / 2, 1 - height / 2)

        annotations.append((0, x_center, y_center, width, height))

    # Random augmentations for variety
    if np.random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, annotations


def generate_dataset(
    output_dir: str,
    num_train: int = 200,
    num_valid: int = 50,
    num_test: int = 50,
    img_size: int = IMG_SIZE,
):
    """Generate full train/valid/test splits with YOLO-format annotations."""
    output_path = Path(output_dir)

    splits = {
        "train": num_train,
        "valid": num_valid,
        "test": num_test,
    }

    total = sum(splits.values())
    generated = 0

    for split_name, count in splits.items():
        img_dir = output_path / split_name / "images"
        lbl_dir = output_path / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Generating {split_name} split ({count} images)...")

        for i in range(count):
            image, annotations = generate_single_image(img_size)

            # Save image
            img_filename = f"debris_{split_name}_{i:04d}.jpg"
            img_path = img_dir / img_filename
            cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save YOLO annotation
            lbl_filename = f"debris_{split_name}_{i:04d}.txt"
            lbl_path = lbl_dir / lbl_filename
            with open(lbl_path, "w") as f:
                for cls_id, xc, yc, w, h in annotations:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            generated += 1
            if (i + 1) % 50 == 0 or (i + 1) == count:
                print(f"  ✅ {i + 1}/{count} images generated")

    print(f"\n{'='*50}")
    print(f"  🎉 Dataset generation complete!")
    print(f"  📊 Total images: {total}")
    print(f"  📂 Output: {output_path.resolve()}")
    print(f"  📝 Format: YOLO (class x_center y_center width height)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Space Debris Dataset")
    parser.add_argument("--output", default=None, help="Output directory (default: dataset/)")
    parser.add_argument("--train", type=int, default=200, help="Number of training images")
    parser.add_argument("--valid", type=int, default=50, help="Number of validation images")
    parser.add_argument("--test", type=int, default=50, help="Number of test images")
    parser.add_argument("--size", type=int, default=640, help="Image size (pixels)")
    args = parser.parse_args()

    # Default output to dataset/ relative to project root
    if args.output is None:
        project_root = Path(__file__).resolve().parent
        output_dir = str(project_root)
    else:
        output_dir = args.output

    generate_dataset(
        output_dir=output_dir,
        num_train=args.train,
        num_valid=args.valid,
        num_test=args.test,
        img_size=args.size,
    )


if __name__ == "__main__":
    main()
