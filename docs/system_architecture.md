# System Architecture

The system consists of four major components.

1. Data Processing
   Satellite images are preprocessed and normalized before entering the model.

2. Debris Detection
   A computer vision model (YOLO or CNN-based detector) identifies debris objects in satellite imagery.

3. Orbit Prediction
   Detected debris objects are tracked across time and their orbital trajectories are predicted.

4. Collision Risk Analysis
   Orbital paths of debris and operational satellites are compared to estimate potential collision risks.

5. Visualization
   A dashboard visualizes orbital movements and potential collision alerts.

Data Flow

Satellite Images
→ Debris Detection Model
→ Orbit Prediction
→ Collision Risk Analysis
→ Visualization Dashboard

