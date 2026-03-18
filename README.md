# 🛰️ Space Debris AI — Intelligent Orbital Debris Detection & Visualization

An AI-powered system that detects space debris from satellite images using YOLOv8 and visualizes debris orbiting Earth through an interactive 3D dashboard.

## Features

- **AI Debris Detection** — YOLOv8-based object detection for space debris in satellite imagery
- **Interactive 3D Earth** — Three.js globe with animated orbital debris, color-coded by risk level
- **Real-time Statistics** — Debris count, confidence scores, and risk analysis
- **Detection History** — Persistent log of all detection sessions
- **Image Upload** — Supports JPG, JPEG, PNG, TIFF formats

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
space-debris-ai/
├── dashboard/          # Streamlit web application
│   └── app.py
├── dataset/            # Training dataset (YOLO format)
│   └── dataset.yaml
├── inference/          # Model inference pipeline
│   └── predict.py
├── models/             # Trained model weights
├── training/           # Model training scripts
│   └── train.py
├── visualization/      # 3D Earth & orbital simulation
│   ├── earth_3d.py
│   └── orbital_simulation.py
├── utils/              # Helper utilities
│   ├── preprocessing.py
│   └── visualization.py
├── logs/               # Detection logs
├── requirements.txt
└── README.md
```

## Training Your Own Model

1. Place labeled satellite images in `dataset/train/`, `dataset/valid/`, `dataset/test/` following YOLO annotation format.
2. Run training:
   ```bash
   python training/train.py
   ```
3. The trained model is saved as `models/debris_detector.pt`.

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| 3D Visualization | Three.js (WebGL) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Image Processing | OpenCV, Pillow |

## License

MIT
