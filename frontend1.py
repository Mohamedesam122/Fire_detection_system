import streamlit as st
import requests
import cv2
import numpy as np
import base64
import os
import time
from PIL import Image
import io

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardianAI · Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — Cyberpunk HUD Dark Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Reset & Root ── */
.stApp {
    background: #05050f !important;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(255,60,0,0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(0,120,255,0.03) 0%, transparent 50%) !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer { visibility: hidden !important; }
header { visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08081a 0%, #0d0d24 40%, #0a0a18 100%) !important;
    border-right: 1px solid rgba(255,80,0,0.15) !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #b8c0d0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Sidebar slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #ff5500 !important;
}

/* ── Typography ── */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: #e8ecf4 !important;
}
.stApp p, .stApp span, .stApp label, .stApp li {
    font-family: 'Rajdhani', sans-serif !important;
    color: #a0a8b8 !important;
    font-size: 1.05rem !important;
}

/* ── Main content ── */
.block-container {
    padding: 1rem 2rem !important;
    max-width: 1500px !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #ff5500 0%, #ff2200 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.25s !important;
    box-shadow: 0 0 20px rgba(255,85,0,0.15) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #ff6a1a 0%, #ff3300 100%) !important;
    box-shadow: 0 0 30px rgba(255,85,0,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(15,15,30,0.8) !important;
    border: 2px dashed rgba(255,85,0,0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(255,85,0,0.5) !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #707888 !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #ff5500, #ff8800) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #05050f; }
::-webkit-scrollbar-thumb { background: #1a1a35; border-radius: 3px; }

/* ═══ Custom Components ═══ */

/* ── Hero Banner ── */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #0a0a1e 0%, #12122e 50%, #0a0a1e 100%);
    border: 1px solid rgba(255,80,0,0.12);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(255,60,0,0.04), inset 0 1px 0 rgba(255,255,255,0.03);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,85,0,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(0,120,255,0.04) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    background: linear-gradient(90deg, #ff5500 0%, #ff8c00 40%, #ff4400 70%, #ff6600 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    position: relative;
    z-index: 1;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: #4a5568;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    margin-top: 6px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}
.hero-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #ff5500, transparent);
    margin-top: 12px;
    position: relative;
    z-index: 1;
}

/* ── HUD Cards ── */
.hud-card {
    background: linear-gradient(145deg, rgba(12,12,28,0.9) 0%, rgba(8,8,20,0.95) 100%);
    border: 1px solid rgba(255,85,0,0.1);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.hud-card:hover {
    border-color: rgba(255,85,0,0.3);
    box-shadow: 0 4px 30px rgba(255,85,0,0.08);
    transform: translateY(-2px);
}
.hud-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,85,0,0.4), transparent);
}
.hud-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.5rem;
}
.hud-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.hud-detail {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.82rem;
    color: #3a4258;
}

.clr-fire   { color: #ff5500; }
.clr-safe   { color: #00e676; }
.clr-warn   { color: #ffab00; }
.clr-info   { color: #448aff; }
.clr-white  { color: #c8d0e0; }

/* ── Alert Panels ── */
.hud-alert {
    border-radius: 10px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.03em;
}
.alert-safe {
    background: rgba(0,230,118,0.06);
    border: 1px solid rgba(0,230,118,0.2);
    color: #00e676;
    box-shadow: 0 0 20px rgba(0,230,118,0.03);
}
.alert-warn {
    background: rgba(255,171,0,0.06);
    border: 1px solid rgba(255,171,0,0.25);
    color: #ffab00;
    box-shadow: 0 0 20px rgba(255,171,0,0.03);
    animation: glow-warn 2s ease-in-out infinite alternate;
}
.alert-danger {
    background: rgba(255,23,68,0.08);
    border: 1px solid rgba(255,23,68,0.3);
    color: #ff1744;
    box-shadow: 0 0 25px rgba(255,23,68,0.06);
    animation: glow-danger 1s ease-in-out infinite alternate;
}
.alert-idle {
    background: rgba(68,138,255,0.05);
    border: 1px solid rgba(68,138,255,0.15);
    color: #5c7caa;
}

@keyframes glow-warn {
    from { box-shadow: 0 0 15px rgba(255,171,0,0.04); }
    to   { box-shadow: 0 0 25px rgba(255,171,0,0.1); }
}
@keyframes glow-danger {
    from { box-shadow: 0 0 15px rgba(255,23,68,0.06); }
    to   { box-shadow: 0 0 35px rgba(255,23,68,0.15); }
}

/* ── Section Headings ── */
.sec-head {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #3a4258;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,85,0,0.1);
    margin: 1.5rem 0 1rem;
}

/* ── Pulse dots ── */
.dot-pulse {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    animation: dot-blink 1.5s infinite;
}
.dot-green  { background: #00e676; box-shadow: 0 0 8px rgba(0,230,118,0.4); }
.dot-red    { background: #ff1744; box-shadow: 0 0 8px rgba(255,23,68,0.4); }
.dot-orange { background: #ff8c00; box-shadow: 0 0 8px rgba(255,140,0,0.4); }
@keyframes dot-blink {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.3; }
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem;
    border-bottom: 1px solid rgba(255,85,0,0.1);
    margin-bottom: 1rem;
}
.sidebar-brand-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 4px;
}
.sidebar-brand-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    background: linear-gradient(90deg, #ff5500, #ff8c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-brand-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3a4258;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─── Backend health check ──────────────────────────────────────────────────
def check_backend_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

backend_online, health_data = check_backend_health()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🔥</span>
        <div class="sidebar-brand-name">GUARDIAN AI</div>
        <div class="sidebar-brand-tag">fire detection system v2.0</div>
    </div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "NAVIGATION",
    ["📊 System Dashboard", "🖼️ Analyze Image", "📹 Analyze Video", "📷 Live Camera Feed"]
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sec-head">Model Configuration</div>', unsafe_allow_html=True)
conf_threshold = st.sidebar.slider(
    "Inference Confidence",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="Higher = fewer false positives, lower = catches more."
)

st.sidebar.markdown('<div class="sec-head">System Status</div>', unsafe_allow_html=True)
if backend_online:
    model_name = health_data.get("model", "YOLOv8")
    classes = list(health_data.get("classes", {}).values())
    st.sidebar.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span class="dot-pulse dot-green"></span>
            <span style="color:#00e676;font-weight:700;font-family:'Rajdhani',sans-serif">ENGINE ONLINE</span>
        </div>
        <div style="font-size:0.82rem;color:#3a4258;font-family:'Share Tech Mono',monospace">
            MODEL: {model_name}<br>
            CLASSES: {classes}
        </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span class="dot-pulse dot-red"></span>
            <span style="color:#ff1744;font-weight:700;font-family:'Rajdhani',sans-serif">ENGINE OFFLINE</span>
        </div>
        <div style="font-size:0.82rem;color:#3a4258;font-family:'Share Tech Mono',monospace">
            Run: python backend1.py
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">GUARDIAN AI</div>
        <div class="hero-sub">Real-time computer vision · fire &amp; thermal hazard detection</div>
        <div class="hero-line"></div>
    </div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if nav_selection == "📊 System Dashboard":
    st.markdown('<div class="sec-head">System Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="hud-card">
                <div class="hud-label">Detection Engine</div>
                <div class="hud-value clr-fire">YOLOv8</div>
                <div class="hud-detail">Custom trained · Roboflow Fire Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        color = "clr-safe" if backend_online else "clr-fire"
        status = "ONLINE" if backend_online else "OFFLINE"
        st.markdown(f"""
            <div class="hud-card">
                <div class="hud-label">Inference Engine</div>
                <div class="hud-value {color}">{status}</div>
                <div class="hud-detail">FastAPI · localhost:8000</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="hud-card">
                <div class="hud-label">Target Anomalies</div>
                <div class="hud-value" style="color:#ff1744">FIRE</div>
                <div class="hud-detail">Single-class thermal detection</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Capabilities</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🎯", "Image Analysis", "Static fire detection on uploaded photos"),
        ("🎬", "Video Pipeline", "Frame-by-frame fire scanning with progress"),
        ("📷", "Live Camera", "Real-time continuous webcam detection"),
        ("⚡", "Fast Inference", "GPU-accelerated YOLOv8 backbone"),
    ]
    for col, (icon, title, desc) in zip([f1,f2,f3,f4], features):
        with col:
            st.markdown(f"""
                <div class="hud-card" style="text-align:center">
                    <div style="font-size:1.8rem;margin-bottom:6px">{icon}</div>
                    <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;color:#c0c8d8;
                                letter-spacing:0.08em;font-weight:600">{title}</div>
                    <div class="hud-detail" style="margin-top:6px">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Quick Start</div>', unsafe_allow_html=True)
    st.markdown("""
    - Select **🖼️ Analyze Image** to test on static photos
    - Select **📹 Analyze Video** to process video recordings
    - Select **📷 Live Camera** for real-time continuous detection
    - Adjust **Inference Confidence** in the sidebar to tune sensitivity
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: IMAGE
# ══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "🖼️ Analyze Image":
    st.markdown('<div class="sec-head">Static Image Inference</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown('<div class="sec-head">Original</div>', unsafe_allow_html=True)
            st.image(file_bytes, use_container_width=True)

        with col_img2:
            st.markdown('<div class="sec-head">Detection Results</div>', unsafe_allow_html=True)

            if not backend_online:
                st.markdown('<div class="hud-alert alert-danger">🚫 Backend offline — cannot run inference</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Scanning for thermal anomalies..."):
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    try:
                        response = requests.post(f"{API_URL}/predict", files=files,
                                                 params={"conf": conf_threshold}, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            count = data.get("count", 0)
                            detections = data.get("detections", [])
                            encoded_image = data.get("image", "")

                            if count == 0:
                                st.markdown('<div class="hud-alert alert-safe">🛡️ SAFE — No fire detected</div>',
                                            unsafe_allow_html=True)
                            elif count <= 2:
                                st.markdown(f'<div class="hud-alert alert-warn">⚠️ WARNING — {count} fire zone(s)</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="hud-alert alert-danger">🚨 CRITICAL — {count} fire zone(s)!</div>',
                                            unsafe_allow_html=True)

                            img_data = base64.b64decode(encoded_image)
                            st.image(img_data, use_container_width=True)

                            if count > 0:
                                st.markdown('<div class="sec-head">Detections</div>', unsafe_allow_html=True)
                                for i, det in enumerate(detections):
                                    st.markdown(f"""
                                        <div class="hud-card" style="margin-bottom:0.5rem;padding:0.8rem 1rem">
                                            <span style="color:#ff5500;font-family:'Orbitron',sans-serif;font-size:0.75rem;font-weight:700">
                                                ZONE {i+1}
                                            </span>
                                            <span style="color:#606878;margin:0 8px">·</span>
                                            <span style="color:#a0a8b8;font-family:'Rajdhani',sans-serif">
                                                {det['class_name']} — {det['confidence']:.1%} confidence
                                            </span>
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "📹 Analyze Video":
    st.markdown('<div class="sec-head">Video Inference Pipeline</div>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"],
                                      label_visibility="collapsed")

    if uploaded_video is not None:
        if not backend_online:
            st.markdown('<div class="hud-alert alert-danger">🚫 Backend offline</div>',
                        unsafe_allow_html=True)
        else:
            temp_filename = "temp_uploaded_video.mp4"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_filename)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f'<div class="hud-card"><div class="hud-label">Total Frames</div><div class="hud-value clr-info">{total_frames}</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="hud-card"><div class="hud-label">Source FPS</div><div class="hud-value clr-info">{fps:.1f}</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="hud-card"><div class="hud-label">Confidence</div><div class="hud-value clr-warn">{conf_threshold:.0%}</div></div>', unsafe_allow_html=True)

            run_btn = st.button("🚀 START INFERENCE PIPELINE", use_container_width=True)

            if run_btn:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                frame_placeholder = st.empty()

                fire_frames_detected = 0
                frame_count = 0
                start_time = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    h, w, _ = frame.shape
                    if w > 640:
                        frame_resized = cv2.resize(frame, (640, int(h * (640 / w))))
                    else:
                        frame_resized = frame.copy()

                    _, encoded_frame = cv2.imencode('.jpg', frame_resized)
                    frame_bytes = encoded_frame.tobytes()

                    try:
                        files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
                        response = requests.post(f"{API_URL}/predict", files=files,
                                                 params={"conf": conf_threshold}, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            count = data.get("count", 0)
                            encoded_image = data.get("image", "")

                            if count > 0:
                                fire_frames_detected += 1

                            img_data = base64.b64decode(encoded_image)
                            frame_placeholder.image(img_data, use_container_width=True)

                        progress_bar.progress(min(1.0, frame_count / total_frames))
                        status_txt.markdown(f"""
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:#4a5568">
                                FRAME {frame_count}/{total_frames} · FIRE ALERTS: {fire_frames_detected}
                            </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error on frame {frame_count}: {e}")
                        break

                cap.release()
                if os.path.exists(temp_filename):
                    try: os.remove(temp_filename)
                    except: pass

                elapsed = time.time() - start_time
                st.markdown(f"""
                    <div class="hud-alert alert-safe">
                        ✅ Complete — {elapsed:.1f}s elapsed · Fire in {fire_frames_detected}/{total_frames} frames
                    </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LIVE CAMERA
# ══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "📷 Live Camera Feed":
    st.markdown('<div class="sec-head">Live Camera Inference</div>', unsafe_allow_html=True)

    if not backend_online:
        st.markdown('<div class="hud-alert alert-danger">🚫 Backend offline — start backend1.py first</div>',
                    unsafe_allow_html=True)
    else:
        col_feed, col_info = st.columns([2.2, 1])

        with col_info:
            st.markdown('<div class="sec-head">Controls</div>', unsafe_allow_html=True)
            start_btn = st.button("▶️  START DETECTION", use_container_width=True, key="start_live")
            stop_btn  = st.button("⏹️  STOP", use_container_width=True, key="stop_live")
            alert_ph = st.empty()
            stats_ph = st.empty()
            alert_ph.markdown('<div class="hud-alert alert-idle">📷 Press START to begin</div>',
                              unsafe_allow_html=True)

        with col_feed:
            frame_ph = st.empty()

        if "live_running" not in st.session_state:
            st.session_state.live_running = False

        if stop_btn:
            st.session_state.live_running = False
        if start_btn:
            st.session_state.live_running = True

        if st.session_state.live_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam.")
                st.session_state.live_running = False
            else:
                prev_time = time.time()
                total_fire = 0
                total_frames = 0

                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    total_frames += 1
                    h, w, _ = frame.shape
                    if w > 640:
                        fr = cv2.resize(frame, (640, int(h * (640 / w))))
                    else:
                        fr = frame.copy()

                    _, enc = cv2.imencode('.jpg', fr)
                    fb = enc.tobytes()

                    try:
                        resp = requests.post(f"{API_URL}/predict",
                                             files={"file": ("f.jpg", fb, "image/jpeg")},
                                             params={"conf": conf_threshold}, timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            count = data.get("count", 0)
                            img64 = data.get("image", "")

                            now = time.time()
                            fps = 1.0 / (now - prev_time + 1e-8)
                            prev_time = now

                            if count > 0:
                                total_fire += 1

                            frame_ph.image(base64.b64decode(img64), use_container_width=True)

                            if count == 0:
                                alert_ph.markdown('<div class="hud-alert alert-safe">🛡️ SAFE — No fire</div>',
                                                  unsafe_allow_html=True)
                            elif count <= 2:
                                alert_ph.markdown(f'<div class="hud-alert alert-warn">⚠️ WARNING — {count} fire zone(s)</div>',
                                                  unsafe_allow_html=True)
                            else:
                                alert_ph.markdown(f'<div class="hud-alert alert-danger">🚨 CRITICAL — {count} fires!</div>',
                                                  unsafe_allow_html=True)

                            stats_ph.markdown(f"""
                                <div class="hud-card" style="margin-top:1rem">
                                    <div class="hud-label">Live Telemetry</div>
                                    <div style="margin-top:8px;font-family:'Share Tech Mono',monospace;
                                                font-size:0.85rem;color:#607088;line-height:1.8">
                                        FPS ............ <span style="color:#00e676">{fps:.1f}</span><br>
                                        FIRES NOW ...... <span style="color:#ff5500">{count}</span><br>
                                        FRAME .......... <span style="color:#448aff">{total_frames}</span><br>
                                        FIRE FRAMES .... <span style="color:#ff1744">{total_fire}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        frame_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        frame_ph.image(frame_rgb, use_container_width=True)
                        alert_ph.markdown(f'<div class="hud-alert alert-danger">⚠️ Error: {e}</div>',
                                          unsafe_allow_html=True)

                cap.release()
                st.session_state.live_running = False
                st.markdown('<div class="hud-alert alert-idle">📷 Camera stopped</div>',
                            unsafe_allow_html=True)
