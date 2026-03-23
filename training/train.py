"""
Space Debris AI — Training Pipeline
Trains a YOLOv8 model on labeled satellite debris imagery.
Saves training metrics and evaluation results for the dashboard.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def train_model(
    data_yaml: str = "dataset/dataset.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    learning_rate: float = 0.001,
    optimizer: str = "AdamW",
    output_dir: str = "models",
):
    """Train YOLOv8 on the space debris dataset and export metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_yaml
    output_path = project_root / output_dir
    logs_path = project_root / "logs"

    if not data_path.exists():
        print(f"Error: Dataset config not found at {data_path}")
        print("Please ensure your dataset is set up in the dataset/ directory.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  🛰️  Space Debris AI — Training Pipeline")
    print("=" * 60)
    print(f"  Model:         {model_name}")
    print(f"  Dataset:       {data_path}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Image Size:    {image_size}")
    print(f"  Optimizer:     {optimizer}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Output Dir:    {output_path}")
    print("=" * 60)

    # Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        optimizer=optimizer,
        lr0=learning_rate,
        project=str(output_path),
        name="debris_detection",
        exist_ok=True,
        verbose=True,
        plots=True,  # Generate confusion matrix and P/R curves
    )

    # Export best model
    best_model_path = output_path / "debris_detection" / "weights" / "best.pt"
    final_model_path = output_path / "debris_detector.pt"

    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"\n✅ Trained model saved to: {final_model_path}")
    else:
        print("\n⚠️  Training completed but best.pt not found.")

    # ─── Save Training Metrics to JSON ─────────────────────
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
        },
        "plots": {
            "confusion_matrix": str(Path(output_dir) / "debris_evaluation" / "confusion_matrix.png"),
            "results": str(Path(output_dir) / "debris_detection" / "results.png")
        },
        "results": {},
    }

    if hasattr(results, "results_dict"):
        metrics["results"] = {
            k: round(float(v), 4) for k, v in results.results_dict.items()
        }

    # ─── Run Validation for detailed metrics ───────────────
    print("\n📊 Running model validation...")
    try:
        val_results = model.val(
            data=str(data_path),
            split="test",
            project=str(output_path),
            name="debris_evaluation",
            exist_ok=True,
            plots=True,  # Generates confusion_matrix.png, PR_curve.png, etc.
        )

        if hasattr(val_results, "box"):
            box = val_results.box
            metrics["evaluation"] = {
                "mAP50": round(float(box.map50), 4),
                "mAP50_95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
                "f1_score": round(
                    2 * (float(box.mp) * float(box.mr))
                    / max(float(box.mp) + float(box.mr), 1e-6),
                    4,
                ),
            }
            print(f"  mAP@50:    {metrics['evaluation']['mAP50']}")
            print(f"  mAP@50-95: {metrics['evaluation']['mAP50_95']}")
            print(f"  Precision: {metrics['evaluation']['precision']}")
            print(f"  Recall:    {metrics['evaluation']['recall']}")
            print(f"  F1 Score:  {metrics['evaluation']['f1_score']}")
    except Exception as e:
        print(f"  ⚠️ Validation step failed: {e}")

    # ─── Save plots paths ──────────────────────────────────
    train_dir = output_path / "debris_detection"
    eval_dir = output_path / "debris_evaluation"

    plot_files = {}
    for name, search_dirs in [
        ("confusion_matrix", [eval_dir, train_dir]),
        ("PR_curve", [eval_dir, train_dir]),
        ("P_curve", [eval_dir, train_dir]),
        ("R_curve", [eval_dir, train_dir]),
        ("F1_curve", [eval_dir, train_dir]),
        ("results", [train_dir]),
    ]:
        for search_dir in search_dirs:
            png_path = search_dir / f"{name}.png"
            if png_path.exists():
                plot_files[name] = str(png_path)
                break

    metrics["plots"] = plot_files

    # Save metrics JSON
    metrics_path = logs_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📋 Metrics saved to: {metrics_path}")

    print("\n📊 Training Results:")
    if hasattr(results, "results_dict"):
        for key, value in results.results_dict.items():
            print(f"  {key}: {value:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Space Debris Detection Model")
    parser.add_argument("--data", default="dataset/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer")
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
