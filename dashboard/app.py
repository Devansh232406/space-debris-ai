"""
Space Debris AI — Streamlit Dashboard
Main application entry point.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

# ─── Path Setup ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict import detect_debris, compute_statistics
from utils.visualization import draw_detections, get_risk_level, get_risk_color_hex
from visualization.orbital_simulation import (
    generate_debris_catalog,
    get_risk_distribution,
    get_altitude_distribution,
    get_collision_zones,
)
from visualization.earth_3d import generate_earth_html

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Space Debris Detection AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: #1a1f35;
        --text-primary: #e0e8ff;
        --text-secondary: #8892b0;
        --accent-blue: #00b4ff;
        --accent-cyan: #00e5ff;
        --accent-purple: #7c3aed;
        --risk-high: #ff4444;
        --risk-medium: #ffaa00;
        --risk-low: #22dd66;
    }

    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 30px 20px 10px;
        background: linear-gradient(180deg, rgba(0,180,255,0.08) 0%, transparent 100%);
        border-bottom: 1px solid rgba(0,180,255,0.1);
        margin-bottom: 24px;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00b4ff, #00e5ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #6b7db3;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Section headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 20px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0,180,255,0.2);
        letter-spacing: 0.5px;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(26,31,53,0.9), rgba(17,24,39,0.9));
        border: 1px solid rgba(0,180,255,0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0,180,255,0.4);
        box-shadow: 0 4px 20px rgba(0,180,255,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00e5ff;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7db3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .risk-high { background: rgba(255,68,68,0.2); color: #ff4444; border: 1px solid rgba(255,68,68,0.3); }
    .risk-medium { background: rgba(255,170,0,0.2); color: #ffaa00; border: 1px solid rgba(255,170,0,0.3); }
    .risk-low { background: rgba(34,221,102,0.2); color: #22dd66; border: 1px solid rgba(34,221,102,0.3); }

    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed rgba(0,180,255,0.3) !important;
        border-radius: 12px !important;
        background: rgba(17,24,39,0.5) !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    }

    /* Plotly chart background fix */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00b4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(0,180,255,0.3);
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(26,31,53,0.5);
        border-radius: 8px;
        color: #8892b0;
        border: 1px solid rgba(0,180,255,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,180,255,0.15) !important;
        color: #00d4ff !important;
        border-color: rgba(0,180,255,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "debris_catalog" not in st.session_state:
    st.session_state.debris_catalog = generate_debris_catalog(150)


# ─── Header ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛰️ Space Debris Detection AI</h1>
    <p>Intelligent Orbital Debris Detection & Visualization System</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_path = st.text_input(
        "Model Path",
        value="models/debris_detector.pt",
        help="Path to trained YOLO model. Demo mode used if not found."
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    st.markdown("---")
    st.markdown("### 🌐 3D Visualization")
    debris_count = st.slider(
        "Debris Objects to Simulate",
        min_value=20,
        max_value=300,
        value=150,
        step=10,
    )
    if st.button("🔄 Regenerate Debris"):
        st.session_state.debris_catalog = generate_debris_catalog(debris_count)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    catalog = st.session_state.debris_catalog
    risk_dist = get_risk_distribution(catalog)
    st.markdown(f"""
    <div style="line-height: 2;">
        <span style="color:#ff4444">●</span> High Risk: <strong>{risk_dist['High']}</strong><br>
        <span style="color:#ffaa00">●</span> Medium Risk: <strong>{risk_dist['Medium']}</strong><br>
        <span style="color:#22dd66">●</span> Low Risk: <strong>{risk_dist['Low']}</strong><br>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content Tabs ──────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Detection",
    "🌍 3D Orbital View",
    "📊 Analytics",
    "📋 History",
    "🧠 Model Performance",
])

# ═══════════════════════════════════════════════════════════
# TAB 1: Image Upload & Detection
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📡 Upload Satellite Image for Debris Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png", "tiff"],
        help="Upload a space/satellite image for debris detection",
    )

    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col_orig, col_detect = st.columns(2)

        with col_orig:
            st.markdown("**📷 Original Image**")
            st.image(image_rgb, use_container_width=True)

        # Run detection
        model_full_path = str(PROJECT_ROOT / model_path)
        detections = detect_debris(
            image,
            model_path=model_full_path,
            conf_threshold=conf_threshold,
            use_demo=not os.path.exists(model_full_path),
        )
        stats = compute_statistics(detections)

        # Draw detections
        annotated = draw_detections(image, detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col_detect:
            st.markdown("**🎯 Detection Results**")
            st.image(annotated_rgb, use_container_width=True)

        # ── Metrics Row ──
        st.markdown('<div class="section-header">📈 Detection Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        risk_class = stats['risk_level'].lower()
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Debris</div>
                <div class="metric-value">{stats['total_debris']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{stats['avg_confidence']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Confidence</div>
                <div class="metric-value">{stats['max_confidence']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value"><span class="risk-badge risk-{risk_class}">{stats['risk_level']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Detection Details Table ──
        if detections:
            st.markdown('<div class="section-header">🔎 Detection Details</div>', unsafe_allow_html=True)
            det_df = pd.DataFrame([
                {
                    "ID": f"DEB-{i+1:03d}",
                    "Label": d["label"],
                    "Confidence": f"{d['confidence']:.2f}",
                    "Bbox (x1,y1,x2,y2)": f"({d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]})",
                }
                for i, d in enumerate(detections)
            ])
            st.dataframe(det_df, use_container_width=True, hide_index=True)

        # ── Confidence Distribution Chart ──
        if detections:
            st.markdown('<div class="section-header">📊 Confidence Distribution</div>', unsafe_allow_html=True)
            conf_values = [d["confidence"] for d in detections]
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Bar(
                x=[f"DEB-{i+1:03d}" for i in range(len(conf_values))],
                y=conf_values,
                marker=dict(
                    color=conf_values,
                    colorscale=[[0, '#22dd66'], [0.5, '#ffaa00'], [1, '#ff4444']],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}<extra></extra>",
            ))
            fig_conf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.5)",
                xaxis_title="Debris ID",
                yaxis_title="Confidence",
                height=320,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # ── Save to history ──
        st.session_state.detection_history.append({
            "Image": uploaded_file.name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Debris Count": stats["total_debris"],
            "Avg Confidence": stats["avg_confidence"],
            "Risk Level": stats["risk_level"],
        })

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#6b7db3;">
            <div style="font-size:4rem; margin-bottom:16px;">🛰️</div>
            <div style="font-size:1.2rem; font-weight:600; color:#8892b0; margin-bottom:8px;">
                Upload a satellite image to begin detection
            </div>
            <div style="font-size:0.9rem;">
                Supported formats: JPG, JPEG, PNG, TIFF
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: 3D Orbital Visualization
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🌍 Interactive 3D Orbital Debris Visualization</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#6b7db3; margin-bottom:16px; font-size:0.9rem;">
        🖱️ <strong>Drag</strong> to rotate &nbsp;|&nbsp; 🔍 <strong>Scroll</strong> to zoom &nbsp;|&nbsp;
        👆 <strong>Hover</strong> debris for details
    </div>
    """, unsafe_allow_html=True)

    # Generate and embed the 3D visualization
    earth_html = generate_earth_html(st.session_state.debris_catalog, width=1100, height=700)
    components.html(earth_html, height=720, scrolling=False)

    # Orbital stats below the globe
    st.markdown('<div class="section-header">🛤️ Orbital Distribution</div>', unsafe_allow_html=True)

    catalog = st.session_state.debris_catalog
    alt_dist = get_altitude_distribution(catalog)
    risk_dist = get_risk_distribution(catalog)

    c1, c2 = st.columns(2)

    with c1:
        fig_alt = go.Figure(data=[go.Pie(
            labels=list(alt_dist.keys()),
            values=list(alt_dist.values()),
            hole=0.5,
            marker=dict(colors=['#00b4ff', '#7c3aed', '#00e5ff']),
            textinfo='label+value',
            textfont=dict(color='white', size=13),
        )])
        fig_alt.update_layout(
            title=dict(text="Orbit Type Distribution", font=dict(color='#00d4ff', size=15)),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=50, b=20),
            legend=dict(font=dict(color='#8892b0')),
        )
        st.plotly_chart(fig_alt, use_container_width=True)

    with c2:
        fig_risk = go.Figure(data=[go.Pie(
            labels=list(risk_dist.keys()),
            values=list(risk_dist.values()),
            hole=0.5,
            marker=dict(colors=['#ff4444', '#ffaa00', '#22dd66']),
            textinfo='label+value',
            textfont=dict(color='white', size=13),
        )])
        fig_risk.update_layout(
            title=dict(text="Risk Level Distribution", font=dict(color='#00d4ff', size=15)),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=50, b=20),
            legend=dict(font=dict(color='#8892b0')),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Collision Zones
    zones = get_collision_zones(catalog)
    if zones:
        st.markdown('<div class="section-header">⚠️ Collision Risk Zones</div>', unsafe_allow_html=True)
        zone_df = pd.DataFrame(zones)
        st.dataframe(zone_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: Analytics Dashboard
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Debris Analytics Dashboard</div>', unsafe_allow_html=True)

    catalog = st.session_state.debris_catalog

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    altitudes = [d["altitude"] for d in catalog]
    velocities = [d["velocity"] for d in catalog]
    sizes = [d["size"] for d in catalog]

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Tracked</div>
            <div class="metric-value">{len(catalog)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Altitude</div>
            <div class="metric-value">{np.mean(altitudes):.0f} km</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Velocity</div>
            <div class="metric-value">{np.mean(velocities):.1f} km/s</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        risk_dist = get_risk_distribution(catalog)
        overall_risk = "High" if risk_dist["High"] > len(catalog) * 0.3 else (
            "Medium" if risk_dist["Medium"] > len(catalog) * 0.3 else "Low"
        )
        risk_class = overall_risk.lower()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Risk</div>
            <div class="metric-value"><span class="risk-badge risk-{risk_class}">{overall_risk}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Altitude distribution histogram
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=altitudes,
            nbinsx=30,
            marker=dict(
                color='rgba(0,180,255,0.6)',
                line=dict(color='rgba(0,180,255,0.8)', width=1),
            ),
            hovertemplate="Altitude: %{x:.0f} km<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.update_layout(
            title=dict(text="Altitude Distribution", font=dict(color='#00d4ff', size=15)),
            xaxis_title="Altitude (km)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.5)",
            height=350,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_scatter = go.Figure()
        for risk in ["High", "Medium", "Low"]:
            filtered = [d for d in catalog if d["risk_level"] == risk]
            color = {"High": "#ff4444", "Medium": "#ffaa00", "Low": "#22dd66"}[risk]
            fig_scatter.add_trace(go.Scatter(
                x=[d["altitude"] for d in filtered],
                y=[d["velocity"] for d in filtered],
                mode="markers",
                name=risk,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Alt: %{x:.0f} km<br>Vel: %{y:.2f} km/s<extra></extra>",
            ))
        fig_scatter.update_layout(
            title=dict(text="Altitude vs Velocity", font=dict(color='#00d4ff', size=15)),
            xaxis_title="Altitude (km)",
            yaxis_title="Velocity (km/s)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.5)",
            height=350,
            margin=dict(t=50, b=40),
            legend=dict(font=dict(color='#8892b0')),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Size distribution
    fig_size = go.Figure()
    fig_size.add_trace(go.Histogram(
        x=sizes,
        nbinsx=25,
        marker=dict(
            color='rgba(124,58,237,0.6)',
            line=dict(color='rgba(124,58,237,0.8)', width=1),
        ),
    ))
    fig_size.update_layout(
        title=dict(text="Debris Size Distribution", font=dict(color='#00d4ff', size=15)),
        xaxis_title="Size (cm)",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.5)",
        height=300,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_size, use_container_width=True)

    # Full debris catalog table
    st.markdown('<div class="section-header">📋 Full Debris Catalog</div>', unsafe_allow_html=True)
    catalog_df = pd.DataFrame(catalog)[["id", "name", "altitude", "velocity", "size", "risk_level", "orbit_type", "inclination"]]
    catalog_df.columns = ["ID", "Name", "Altitude (km)", "Velocity (km/s)", "Size (cm)", "Risk", "Orbit", "Inclination (°)"]
    st.dataframe(catalog_df, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════
# TAB 4: Detection History
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📋 Detection History Log</div>', unsafe_allow_html=True)

    if st.session_state.detection_history:
        history_df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # History analytics
        st.markdown('<div class="section-header">📈 Detection Trends</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=history_df["Image"],
                y=history_df["Debris Count"],
                marker=dict(color='rgba(0,180,255,0.7)'),
            ))
            fig_trend.update_layout(
                title=dict(text="Debris Count per Image", font=dict(color='#00d4ff', size=14)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.5)",
                height=300,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c2:
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Scatter(
                x=history_df["Image"],
                y=history_df["Avg Confidence"],
                mode="lines+markers",
                marker=dict(color='#00e5ff', size=8),
                line=dict(color='#00e5ff', width=2),
            ))
            fig_conf.update_layout(
                title=dict(text="Avg Confidence per Image", font=dict(color='#00d4ff', size=14)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.5)",
                height=300,
                margin=dict(t=50, b=40),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#6b7db3;">
            <div style="font-size:3rem; margin-bottom:16px;">📋</div>
            <div style="font-size:1.1rem; font-weight:500; color:#8892b0;">
                No detection history yet. Upload and analyze images in the Detection tab.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 5: Model Performance
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🧠 Model Performance & Evaluation</div>', unsafe_allow_html=True)

    metrics_path = PROJECT_ROOT / "logs" / "training_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # ── Training Config ──
        config = metrics.get("config", {})
        st.markdown('<div class="section-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value" style="font-size:1.2rem">{config.get('model', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Epochs</div>
                <div class="metric-value">{config.get('epochs', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Batch Size</div>
                <div class="metric-value">{config.get('batch_size', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Image Size</div>
                <div class="metric-value">{config.get('image_size', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Evaluation Metrics ──
        evaluation = metrics.get("evaluation", {})
        if evaluation:
            st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            metric_items = [
                (m1, "mAP@50", evaluation.get("mAP50", 0), "#00e5ff"),
                (m2, "mAP@50-95", evaluation.get("mAP50_95", 0), "#7c3aed"),
                (m3, "Precision", evaluation.get("precision", 0), "#22dd66"),
                (m4, "Recall", evaluation.get("recall", 0), "#ffaa00"),
                (m5, "F1 Score", evaluation.get("f1_score", 0), "#ff4444"),
            ]
            for col, label, value, color in metric_items:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color}">{value:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            # Metrics bar chart
            fig_metrics = go.Figure()
            metric_names = list(evaluation.keys())
            metric_values = list(evaluation.values())
            colors = ["#00e5ff", "#7c3aed", "#22dd66", "#ffaa00", "#ff4444"]
            fig_metrics.add_trace(go.Bar(
                x=metric_names,
                y=metric_values,
                marker=dict(color=colors[:len(metric_names)]),
                text=[f"{v:.4f}" for v in metric_values],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_metrics.update_layout(
                title=dict(text="Model Evaluation Metrics", font=dict(color="#00d4ff", size=15)),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.5)",
                height=350,
                margin=dict(t=50, b=40),
                yaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        # ── Training Plots ──
        plots = metrics.get("plots", {})
        if plots:
            st.markdown('<div class="section-header">📊 Training & Evaluation Plots</div>', unsafe_allow_html=True)

            # Show plots in 2-column grid
            plot_names = {
                "confusion_matrix": "Confusion Matrix",
                "PR_curve": "Precision-Recall Curve",
                "P_curve": "Precision Curve",
                "R_curve": "Recall Curve",
                "F1_curve": "F1 Score Curve",
                "results": "Training Results",
            }

            available_plots = [(k, v) for k, v in plots.items() if os.path.exists(v)]

            for i in range(0, len(available_plots), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(available_plots):
                        key, path = available_plots[idx]
                        with col:
                            display_name = plot_names.get(key, key)
                            st.markdown(f"**{display_name}**")
                            st.image(path, use_container_width=True)

        # ── Training Timestamp ──
        timestamp = metrics.get("timestamp", "Unknown")
        st.markdown(f"""
        <div style="text-align:center; color:#4a5578; padding:20px; font-size:0.85rem;">
            Last trained: {timestamp} &nbsp;|&nbsp;
            Optimizer: {config.get('optimizer', 'N/A')} &nbsp;|&nbsp;
            LR: {config.get('learning_rate', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#6b7db3;">
            <div style="font-size:4rem; margin-bottom:16px;">🧠</div>
            <div style="font-size:1.2rem; font-weight:600; color:#8892b0; margin-bottom:8px;">
                No trained model metrics found
            </div>
            <div style="font-size:0.9rem;">
                Train a model using <code>python training/train.py</code> to see performance metrics here.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5578; padding:10px; font-size:0.8rem;">
    🛰️ Space Debris Detection AI &nbsp;|&nbsp; Powered by YOLOv8 + Three.js &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
