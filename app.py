"""
FireGuard AI – Streamlit Dashboard  (Standalone Mode)
=====================================================
Run:
    streamlit run app.py
"""

import io
import os
import sqlite3
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000"
DB_PATH = "fireguard.db"

# ─── Load model ONCE at startup ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ─── Local DB helpers ───────────────────────────────────────────────────────
def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            source_type TEXT    NOT NULL,
            fire_count  INTEGER NOT NULL,
            max_conf    REAL    NOT NULL,
            avg_conf    REAL    NOT NULL,
            fps         REAL
        )
    """)
    con.commit()
    con.close()

_init_db()

def _log_local(source_type, fire_count, max_conf, avg_conf, fps=None):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO detections (timestamp, source_type, fire_count, max_conf, avg_conf, fps) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), source_type, fire_count,
         round(max_conf, 4), round(avg_conf, 4), fps),
    )
    con.commit()
    con.close()

def _get_local_history(limit=100):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, timestamp, source_type, fire_count, max_conf, avg_conf, fps "
        "FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [{"id": r[0], "timestamp": r[1], "source_type": r[2],
             "fire_count": r[3], "max_conf": r[4], "avg_conf": r[5], "fps": r[6]}
            for r in rows]

def _get_local_stats():
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT COUNT(*), SUM(fire_count), AVG(max_conf), MAX(timestamp) "
        "FROM detections"
    ).fetchone()
    con.close()
    return {
        "total_detections": row[0] or 0,
        "total_fires":      row[1] or 0,
        "avg_confidence":   round(row[2] or 0, 4),
        "most_recent":      row[3],
    }

def _clear_local_history():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM detections")
    con.commit()
    con.close()


# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background-color: #0a0a1a !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #0a0a1a !important;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #0a0a1a !important;
}

/* Force all text to light colour */
.stApp p, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #e2e8f0 !important;
}

/* Hide Streamlit top bar & footer */
#MainMenu       { visibility: hidden !important; }
footer           { visibility: hidden !important; }
header           { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Hide sidebar completely since we don't use it */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Block container */
.block-container {
    padding: 1rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Tabs styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background-color: #1e1e3f;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background-color: #1a1a3e !important;
    border: 2px dashed #2d2d5e !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] * { color: #94a3b8 !important; }

/* ── Slider ── */
[data-testid="stSlider"] label p { color: #c9d1e0 !important; }

/* ── Toggle ── */
.stToggle label span p { color: #c9d1e0 !important; }

/* ── Custom cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 0.5rem;
}
.metric-card .label {
    font-size: 0.78rem; color: #8892a4;
    text-transform: uppercase; letter-spacing: 1px;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }

.val-blue   { color: #60a5fa !important; }
.val-green  { color: #34d399 !important; }
.val-orange { color: #fb923c !important; }
.val-red    { color: #f87171 !important; }

/* ── Alert banners ── */
.alert-fire {
    background: linear-gradient(135deg, #3b0000, #7f1d1d);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: pulse-red 1.5s infinite;
}
.alert-safe {
    background: linear-gradient(135deg, #003b1e, #064e3b);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%      { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
}

/* ── Section titles ── */
.section-title {
    font-size: 0.85rem; font-weight: 600; color: #94a3b8 !important;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1rem 0 0.6rem;
    border-bottom: 1px solid #2d2d5e; padding-bottom: 0.3rem;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; width: 100% !important;
    font-weight: 600 !important; transition: all 0.2s !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
}
div.stButton > button:disabled {
    background: #2d2d5e !important;
    color: #6b7280 !important;
}

/* ── Settings bar ── */
.settings-bar {
    background: linear-gradient(135deg, #1e1e3f 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: #2d2d5e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def get_history():
    try:
        r = requests.get(f"{API_URL}/history?limit=100", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return _get_local_history()

def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return _get_local_stats()

def log_detection(source_type, fire_count, max_conf, avg_conf, fps=None):
    _log_local(source_type, fire_count, max_conf, avg_conf, fps)

def run_inference_local(frame_bgr, conf_thresh=0.25):
    results    = model.predict(frame_bgr, conf=conf_thresh, verbose=False)
    boxes      = results[0].boxes
    fire_count = len(boxes)
    confs      = [float(b.conf[0]) for b in boxes] if fire_count else []
    max_conf   = max(confs) if confs else 0.0
    avg_conf   = sum(confs) / len(confs) if confs else 0.0
    detections = [
        {"bbox": [round(v, 2) for v in b.xyxy[0].tolist()],
         "confidence": round(float(b.conf[0]), 4)}
        for b in boxes
    ]
    return {
        "fire_count":     fire_count,
        "max_confidence": max_conf,
        "avg_confidence": avg_conf,
        "detections":     detections,
        "annotated":      results[0].plot(),
    }

def play_alarm():
    st.markdown("""<script>
    const c=new AudioContext(),o=c.createOscillator(),g=c.createGain();
    o.connect(g);g.connect(c.destination);o.type='sawtooth';o.frequency.value=880;
    g.gain.setValueAtTime(0.3,c.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+0.8);
    o.start();o.stop(c.currentTime+0.8);
    </script>""", unsafe_allow_html=True)

def draw_status_bar(frame, fire_count, fps):
    h, w = frame.shape[:2]
    if fire_count > 0:
        cv2.rectangle(frame, (0,0), (w,50), (0,0,180), -1)
        cv2.putText(frame, f"FIRE: {fire_count}  |  FPS: {fps:.1f}",
                    (10,33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    else:
        cv2.rectangle(frame, (0,0), (w,50), (0,120,0), -1)
        cv2.putText(frame, f"ALL CLEAR  |  FPS: {fps:.1f}",
                    (10,33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    return frame

def show_alert(placeholder, fire_count, conf=0.0):
    if fire_count > 0:
        placeholder.markdown(f"""
        <div class="alert-fire">
          <div style="font-size:1.5rem">🔥</div>
          <div style="font-weight:700;color:#fca5a5">FIRE DETECTED</div>
          <div style="font-size:0.8rem;color:#fca5a5">{fire_count} region(s)</div>
        </div>""", unsafe_allow_html=True)
    else:
        placeholder.markdown("""
        <div class="alert-safe">
          <div style="font-size:1.5rem">✅</div>
          <div style="font-weight:700;color:#6ee7b7">ALL CLEAR</div>
        </div>""", unsafe_allow_html=True)

def show_metrics(placeholder, **kwargs):
    html = ""
    colors = {"FPS":"val-green","Fires":"val-red","Frame":"val-blue","Conf":"val-orange"}
    for label, val in kwargs.items():
        c = colors.get(label, "val-blue")
        html += f'<div class="metric-card"><div class="label">{label}</div><div class="value {c}">{val}</div></div>'
    placeholder.markdown(html, unsafe_allow_html=True)

def conf_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value*100,
        number={"suffix":"%","font":{"color":"#e2e8f0","size":28}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#6b7280"},
            "bar":{"color":"#f87171" if value>0.5 else "#34d399"},
            "bgcolor":"#1e1e3f","bordercolor":"#2d2d5e",
            "steps":[{"range":[0,40],"color":"#064e3b"},
                     {"range":[40,70],"color":"#78350f"},
                     {"range":[70,100],"color":"#7f1d1d"}],
            "threshold":{"line":{"color":"#fbbf24","width":3},"thickness":0.75,"value":75},
        },
        title={"text":"Max Confidence","font":{"color":"#94a3b8","size":13}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=200,margin=dict(l=20,r=20,t=30,b=10))
    return fig


# ─── Session state defaults ──────────────────────────────────────────────────
for key, val in [("cam_running", False), ("vid_running", False)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + SETTINGS BAR (replaces sidebar)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:0.5rem 0 0.8rem">
  <span style="font-size:2rem">🔥</span>
  <span style="font-size:1.5rem;font-weight:700;color:#f87171;vertical-align:middle;margin-left:8px">FireGuard AI</span>
  <div style="font-size:0.78rem;color:#64748b;margin-top:2px">Real-Time Fire Detection System · YOLOv8</div>
</div>
""", unsafe_allow_html=True)

# ── Settings bar: slider + toggle in columns ──
set_col1, set_col2, set_col3 = st.columns([2, 1, 1])
with set_col1:
    conf_thresh = st.slider("🎯 Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
with set_col2:
    sound_alarm = st.toggle("🔊 Sound Alarm", value=True)
with set_col3:
    api_ok = check_api()
    if api_ok:
        st.success("🟢 Backend Connected", icon="✅")
    else:
        st.info("🟡 Standalone Mode", icon="ℹ️")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS (replaces sidebar radio buttons)
# ═══════════════════════════════════════════════════════════════════════════════
tab_image, tab_video, tab_webcam, tab_analytics = st.tabs([
    "📷 Image", "🎬 Video", "📹 Webcam", "📊 Analytics"
])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed", key="img_upload")
    if uploaded:
        img_bytes = uploaded.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np    = np.array(pil_img)
        img_bgr   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        col_img, col_res = st.columns([1.5, 1])
        with col_img:
            st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
            preview_ph = st.empty()
            preview_ph.image(pil_img, use_container_width=True)

        with col_res:
            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
            if st.button("🔍 Detect Fire", key="detect_img"):
                with st.spinner("Analysing…"):
                    t0  = time.time()
                    res = run_inference_local(img_bgr, conf_thresh)
                    ms  = (time.time() - t0) * 1000

                fire_count = res["fire_count"]
                max_conf   = res["max_confidence"]
                detections = res["detections"]

                if fire_count > 0:
                    log_detection("image", fire_count, max_conf, res["avg_confidence"])

                alert_ph = st.empty()
                show_alert(alert_ph, fire_count, max_conf)
                if fire_count > 0 and sound_alarm:
                    play_alarm()

                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="label">Fires</div><div class="value val-red">{fire_count}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="label">Confidence</div><div class="value val-orange">{max_conf:.0%}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="label">Speed</div><div class="value val-blue">{ms:.0f}ms</div></div>', unsafe_allow_html=True)

                st.plotly_chart(conf_gauge(max_conf), use_container_width=True, config={"displayModeBar":False})

                if detections:
                    ann = img_np.copy()
                    for d in detections:
                        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                        cv2.rectangle(ann, (x1,y1), (x2,y2), (255,60,60), 2)
                        lbl = f"Fire {d['confidence']:.0%}"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(ann, (x1,y1-th-8), (x1+tw+6,y1), (255,60,60), -1)
                        cv2.putText(ann, lbl, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    preview_ph.image(ann, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<div class="section-title">Upload Video</div>', unsafe_allow_html=True)
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"],
                                    label_visibility="collapsed", key="vid_upload")
    if uploaded_vid:
        col_vid, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
            start_vid = st.button("▶️ Start Detection", use_container_width=True,
                                  disabled=st.session_state.vid_running, key="start_vid")
            stop_vid  = st.button("⏹️ Stop Detection",  use_container_width=True,
                                  disabled=not st.session_state.vid_running, key="stop_vid")
            st.markdown("<br>", unsafe_allow_html=True)
            alert_ph  = st.empty()
            metric_ph = st.empty()
            alert_ph.markdown('<div class="metric-card" style="text-align:center"><div style="font-size:2rem">🎬</div><div style="color:#64748b">Press Start</div></div>', unsafe_allow_html=True)

        with col_vid:
            frame_ph = st.empty()

        if stop_vid:
            st.session_state.vid_running = False

        if start_vid:
            st.session_state.vid_running = True
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.close()

            cap          = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx    = 0
            total_fires  = 0
            fire_frames  = 0
            prev_time    = time.time()
            progress     = st.progress(0, text="Processing…")

            while cap.isOpened() and st.session_state.vid_running:
                ret, frame = cap.read()
                if not ret:
                    break

                res        = run_inference_local(frame, conf_thresh)
                annotated  = res["annotated"]
                fire_count = res["fire_count"]

                now       = time.time()
                fps_live  = 1 / (now - prev_time + 1e-8)
                prev_time = now

                if fire_count > 0:
                    total_fires += fire_count
                    fire_frames += 1

                annotated = draw_status_bar(annotated, fire_count, fps_live)
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                show_alert(alert_ph, fire_count)
                if fire_count > 0 and sound_alarm:
                    play_alarm()

                show_metrics(metric_ph,
                    Frame=f"{frame_idx}/{total_frames}",
                    Fires=total_fires,
                    FPS=f"{fps_live:.1f}")

                frame_idx += 1
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0),
                                  text=f"Frame {frame_idx}/{total_frames}")

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            if total_fires > 0:
                log_detection("video", total_fires, 0.0, 0.0)

            st.session_state.vid_running = False
            st.success(f"✅ Done! Fire detected in {fire_frames}/{total_frames} frames.")


# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    col_cam, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
        start_cam = st.button("▶️ Start Camera", use_container_width=True,
                              disabled=st.session_state.cam_running, key="start_cam")
        stop_cam  = st.button("⏹️ Stop Camera",  use_container_width=True,
                              disabled=not st.session_state.cam_running, key="stop_cam")
        st.markdown("<br>", unsafe_allow_html=True)
        alert_ph_cam  = st.empty()
        metric_ph_cam = st.empty()
        alert_ph_cam.markdown('<div class="metric-card" style="text-align:center"><div style="font-size:2rem">📷</div><div style="color:#64748b">Press Start</div></div>', unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="section-title">Live Feed</div>', unsafe_allow_html=True)
        frame_ph_cam = st.empty()

    if stop_cam:
        st.session_state.cam_running = False

    if start_cam:
        st.session_state.cam_running = True

    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera not found. Make sure a webcam is connected.")
            st.session_state.cam_running = False
        else:
            prev_time = time.time()
            while st.session_state.cam_running:
                ret, frame = cap.read()
                if not ret:
                    break

                res        = run_inference_local(frame, conf_thresh)
                annotated  = res["annotated"]
                fire_count = res["fire_count"]

                now       = time.time()
                fps_live  = 1 / (now - prev_time + 1e-8)
                prev_time = now

                annotated = draw_status_bar(annotated, fire_count, fps_live)
                frame_ph_cam.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                show_alert(alert_ph_cam, fire_count)
                if fire_count > 0 and sound_alarm:
                    play_alarm()

                show_metrics(metric_ph_cam, FPS=f"{fps_live:.1f}", Fires=fire_count)

            cap.release()
            frame_ph_cam.empty()
            st.session_state.cam_running = False


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    stats   = get_stats()
    history = get_history()

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, cls in [
        (k1, "Total Events",   stats.get("total_detections", 0),            "val-blue"),
        (k2, "Total Fires",    stats.get("total_fires", 0),                  "val-red"),
        (k3, "Avg Confidence", f"{stats.get('avg_confidence', 0):.0%}",      "val-orange"),
        (k4, "Last Detection", (stats.get("most_recent") or "—")[:16],       "val-green"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value {cls}" style="font-size:1.4rem">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-title">Fire Detection Timeline</div>', unsafe_allow_html=True)
        if history:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["fire_count"],
                                     fill="tozeroy", line={"color":"#f87171","width":2},
                                     fillcolor="rgba(248,113,113,0.15)"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font={"color":"#94a3b8"},height=220,margin=dict(l=10,r=10,t=10,b=10),
                              xaxis={"gridcolor":"#1e1e3f"},yaxis={"gridcolor":"#1e1e3f"})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No history yet.")

    with ch2:
        st.markdown('<div class="section-title">Source Breakdown</div>', unsafe_allow_html=True)
        if history:
            df_h = pd.DataFrame(history)
            src  = df_h["source_type"].value_counts().reset_index()
            src.columns = ["source", "count"]
            fig2 = px.pie(src, names="source", values="count",
                          color_discrete_sequence=["#f87171","#60a5fa","#34d399"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"#94a3b8"},
                               height=220,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No history yet.")

    st.markdown('<div class="section-title">Detection Log</div>', unsafe_allow_html=True)
    if history:
        df_show = pd.DataFrame(history)[["timestamp","source_type","fire_count","max_conf","avg_conf"]]
        df_show.columns = ["Timestamp","Source","Fires","Max Conf","Avg Conf"]
        df_show["Max Conf"] = df_show["Max Conf"].apply(lambda x: f"{x:.0%}")
        df_show["Avg Conf"] = df_show["Avg Conf"].apply(lambda x: f"{x:.0%}")
        st.dataframe(df_show, use_container_width=True, height=300)
        if st.button("🗑️ Clear History", key="clear_hist"):
            _clear_local_history()
            try:
                requests.delete(f"{API_URL}/history", timeout=3)
            except:
                pass
            st.rerun()
    else:
        st.info("No detection events yet.")
