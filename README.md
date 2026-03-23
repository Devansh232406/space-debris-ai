# 🛰️ Space Debris AI — Intelligent Orbital Debris Detection & Visualization

An AI-powered system that detects space debris from satellite images using **YOLOv8** and visualizes debris orbiting Earth through an interactive **3D dashboard**.

![Space Debris AI Dashboard](https://via.placeholder.com/1000x500.png?text=Space+Debris+AI+Dashboard+-+Insert+Screenshot+Here)

**Live Demo**: [[Streamlit Cloud Deployment Link] ](https://space-debris-ai-ghueh7hcxrn4ued3bhwcdv.streamlit.app/)

---

## ✨ Features

- **🤖 AI Debris Detection** — YOLOv8-based object detection for space debris in satellite imagery
- **🌍 Interactive 3D Earth** — Three.js globe with animated orbital debris, color-coded by risk level
- **📊 Real-time Analytics** — Debris count, confidence scores, risk analysis, altitude/velocity charts
- **🧠 Model Performance Dashboard** — mAP, precision, recall, F1 score, confusion matrix, P/R curves
- **📋 Detection History** — Persistent log of all detection sessions with trend analysis
- **🔬 Synthetic Dataset Generator** — Procedural generation of labeled satellite imagery for training
- **✅ Unit Tests** — 55 tests covering inference, preprocessing, and orbital simulation modules

## 🧠 Deep Learning Pipeline

```
Synthetic Dataset Generator → YOLOv8n Training (30 epochs) → Model Evaluation → Dashboard
       ↓                              ↓                            ↓
  300 labeled images          debris_detector.pt           mAP, P/R curves,
  (YOLO format)               (trained weights)           confusion matrix
```

### Architecture
- **Model**: YOLOv8 Nano (single-stage real-time object detector)
- **Optimizer**: AdamW with learning rate 0.001
- **Input**: 640×640 satellite images
- **Output**: Bounding boxes + confidence scores for debris objects
- **Augmentation**: Random rotation, brightness, contrast, noise, horizontal flip

### Performance Metrics (30 Epochs)
- **mAP@50**: 0.9538
- **mAP@50-95**: 0.6391
- **Precision**: 0.9325
- **Recall**: 0.9044
- **F1 Score**: 0.9182

The model generalizes exceptionally well on our synthetic debris dataset, maintaining over 90% precision and recall for high-confidence object identification.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training dataset
python dataset/generate_dataset.py

# 3. Train the model
python training/train.py --epochs 30

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```
space-debris-ai/
├── dashboard/              # Streamlit web application
│   └── app.py              # Main app (5 tabs: Detection, 3D, Analytics, History, Model)
├── dataset/                # Training dataset (YOLO format)
│   ├── dataset.yaml        # YOLO dataset config
│   ├── generate_dataset.py # Synthetic data generator
│   ├── train/              # Training images + labels
│   ├── valid/              # Validation images + labels
│   └── test/               # Test images + labels
├── inference/              # Model inference pipeline
│   └── predict.py          # Detection with YOLO or demo fallback
├── models/                 # Trained model weights
│   └── debris_detector.pt  # Trained YOLOv8 weights
├── training/               # Model training scripts
│   └── train.py            # Training + evaluation + metrics export
├── visualization/          # 3D Earth & orbital simulation
│   ├── earth_3d.py         # Three.js WebGL globe generator
│   └── orbital_simulation.py
├── utils/                  # Helper utilities
│   ├── preprocessing.py    # Data augmentation pipeline
│   └── visualization.py    # Detection drawing utilities
├── tests/                  # Unit tests (55 tests)
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_orbital_simulation.py
├── logs/                   # Training metrics & detection logs
│   └── training_metrics.json
├── requirements.txt
└── README.md
```

## 🏋️ Training Your Own Model

```bash
# Generate synthetic dataset (customizable count)
python dataset/generate_dataset.py --train 300 --valid 75 --test 75

# Train with custom parameters
python training/train.py --epochs 50 --batch 16 --imgsz 640 --lr 0.001

# Model is saved as models/debris_detector.pt
# Metrics are saved to logs/training_metrics.json
```

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) + PyTorch |
| 3D Visualization | Three.js (WebGL) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |
| Testing | pytest |

## 📄 License

MIT
