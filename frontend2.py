"""
GuardianAI – Fire Detection Dashboard
Run: streamlit run app.py
"""

import base64
import io
import os
import time
import tempfile

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="GuardianAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

for k, v in [("live_running", False), ("vid_running", False),
              ("stop_requested", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080C10;
    color: #C8D0D8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1440px; }
[data-testid="collapsedControl"] { display: none !important; }

[data-testid="stSidebar"] {
    background: #0C1018;
    border-right: 1px solid #1E2530;
}
[data-testid="stSidebar"] * { color: #8A95A0 !important; }
[data-testid="stSidebar"] h2 { color: #E8EDF2 !important; }

.g-card {
    background: #0E1420;
    border: 1px solid #1E2A38;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.6rem;
}
.g-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4A5568;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
}
.g-card-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    line-height: 1;
}

.c-amber  { color: #F59E0B; }
.c-cyan   { color: #22D3EE; }
.c-rose   { color: #FB7185; }
.c-lime   { color: #84CC16; }
.c-violet { color: #A78BFA; }
.c-muted  { color: #4A5568; }

.badge { display:inline-flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;
         font-size:0.78rem;padding:0.3rem 0.8rem;border-radius:3px;font-weight:500;letter-spacing:0.05em; }
.badge-safe   { background:#0A1F14;border:1px solid #166534;color:#4ADE80; }
.badge-danger { background:#1C0A0A;border:1px solid #991B1B;color:#F87171; }

.alert-bar { padding:0.9rem 1.2rem;border-radius:4px;font-family:'DM Mono',monospace;
             font-size:0.85rem;font-weight:500;margin-bottom:1rem;
             display:flex;align-items:center;gap:0.6rem; }
.alert-safe   { background:#0A1F14;border-left:3px solid #4ADE80;color:#4ADE80; }
.alert-warn   { background:#1C1200;border-left:3px solid #FBBF24;color:#FBBF24; }
.alert-danger { background:#1C0A0A;border-left:3px solid #F87171;color:#F87171;
                animation:flash 1.2s infinite; }
.alert-idle   { background:#0E1420;border-left:3px solid #1E2A38;color:#4A5568; }

@keyframes flash {
    0%,100% { border-left-color:#F87171; }
    50%      { border-left-color:#7F1D1D; }
}

.g-divider { border:none;border-top:1px solid #1E2A38;margin:1.2rem 0; }
.g-section { font-family:'DM Mono',monospace;font-size:0.7rem;color:#2D3748;
             text-transform:uppercase;letter-spacing:0.2em;margin-bottom:0.8rem;
             padding-bottom:0.4rem;border-bottom:1px solid #1E2530; }

/* Normal button */
.stButton > button {
    background: #0E1420;
    color: #C8D0D8;
    border: 1px solid #1E2A38;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 0.55rem 1.2rem;
    width: 100%;
    transition: all 0.15s;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    border-color: #22D3EE;
    color: #22D3EE;
    background: #0A1520;
}
.stButton > button:disabled {
    opacity: 0.25;
    cursor: not-allowed;
}

/* Stop button — red tint */
div[data-testid="stButton"]:has(button[kind="secondary"]) > button {
    border-color: #991B1B;
    color: #F87171;
}
div[data-testid="stButton"]:has(button[kind="secondary"]) > button:hover {
    background: #1C0A0A;
    border-color: #F87171;
}

[data-testid="stFileUploader"] {
    background: #0E1420;
    border: 1px dashed #1E2A38;
    border-radius: 6px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080C10; }
::-webkit-scrollbar-thumb { background: #1E2530; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except:
        pass
    return False, {}

def predict_api(img_bytes, conf):
    r = requests.post(
        f"{API_URL}/predict",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
        params={"conf": conf},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def encode_frame(frame, quality=70):
    """Encode frame to JPEG bytes quickly."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def alert_html(count):
    if count == 0:
        return "<div class='alert-bar alert-safe'>🛡 SAFE — No fire detected</div>"
    elif count <= 2:
        return f"<div class='alert-bar alert-warn'>⚠ WARNING — {count} fire zone(s)</div>"
    else:
        return f"<div class='alert-bar alert-danger'>🚨 CRITICAL — {count} fire zone(s)</div>"

def idle_alert():
    return "<div class='alert-bar alert-idle'>○ Waiting...</div>"

def card(label, value, color="c-cyan"):
    return (f"<div class='g-card'>"
            f"<div class='g-card-label'>{label}</div>"
            f"<div class='g-card-value {color}'>{value}</div>"
            f"</div>")

def draw_bar(frame, count, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    color = (20, 20, 140) if count > 0 else (10, 80, 10)
    cv2.rectangle(overlay, (0, 0), (w, 48), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    txt = f"FIRE: {count}  |  {fps:.1f} FPS" if count > 0 else f"CLEAR  |  {fps:.1f} FPS"
    cv2.putText(frame, txt, (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    return frame



api_ok, health = check_api()


with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem;text-align:center;'>
      <div style='font-size:1.6rem;font-weight:800;color:#E8EDF2;letter-spacing:-0.03em;'>
        GUARDIAN<span style='color:#F59E0B;'>AI</span>
      </div>
      <div style='font-size:0.72rem;color:#2D3748;font-family:DM Mono,monospace;
                  letter-spacing:0.15em;margin-top:4px;'>FIRE DETECTION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='g-section'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "⬡  Dashboard",
        "⬡  Image Analysis",
        "⬡  Video Analysis",
        "⬡  Live Camera",
    ], label_visibility="collapsed")

    st.markdown("<hr class='g-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='g-section'>Model Config</div>", unsafe_allow_html=True)
    conf = st.slider("Confidence Threshold", 0.10, 0.90, 0.40, 0.05)

    st.markdown("<hr class='g-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='g-section'>System Status</div>", unsafe_allow_html=True)

    if api_ok:
        st.markdown(f"""
        <div class='badge badge-safe' style='margin-bottom:6px'>● API ONLINE</div><br>
        <span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#2D3748;'>
        MODEL: {health.get('model','YOLOv8')}<br>
        CLASSES: {list(health.get('classes',{}).values())}
        </span>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='badge badge-danger' style='margin-bottom:6px'>● API OFFLINE</div><br>
        <span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#2D3748;'>
        Run backend.py on port 8000
        </span>""", unsafe_allow_html=True)



st.markdown("""
<div style='margin-bottom:2rem;'>
  <div style='font-size:2rem;font-weight:800;color:#E8EDF2;letter-spacing:-0.04em;'>
    GUARDIAN<span style='color:#F59E0B;'>AI</span>
    <span style='font-size:0.9rem;font-weight:400;color:#2D3748;
                 font-family:DM Mono,monospace;margin-left:1rem;'>
      FIRE DETECTION · YOLOv8
    </span>
  </div>
</div>
""", unsafe_allow_html=True)



if page == "⬡  Dashboard":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card("Detection Engine", "YOLOv8", "c-amber"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("API Status",
                         "ONLINE" if api_ok else "OFFLINE",
                         "c-lime" if api_ok else "c-rose"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("Target Class", "FIRE / FLAME", "c-rose"), unsafe_allow_html=True)

    st.markdown("<hr class='g-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='g-section'>How To Use</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.82rem;color:#4A5568;line-height:2;'>
    → <b style='color:#C8D0D8;'>Image Analysis</b> — upload a photo, get instant detection<br>
    → <b style='color:#C8D0D8;'>Video Analysis</b>  — upload a video, watch frame-by-frame detection live<br>
    → <b style='color:#C8D0D8;'>Live Camera</b>     — real-time webcam detection via backend API<br>
    → Adjust <b style='color:#C8D0D8;'>Confidence Threshold</b> in sidebar to tune sensitivity
    </div>
    """, unsafe_allow_html=True)



elif page == "⬡  Image Analysis":
    st.markdown("<div class='g-section'>Static Image Inference</div>", unsafe_allow_html=True)

    if not api_ok:
        st.error("⚠ Backend API is offline. Start backend.py first.")
        st.stop()

    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp"],
                                label_visibility="collapsed")

    if uploaded:
        img_bytes = uploaded.read()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("<div class='g-section'>Original</div>", unsafe_allow_html=True)
            st.image(img_bytes, use_container_width=True)

        with col_r:
            st.markdown("<div class='g-section'>Detection Result</div>", unsafe_allow_html=True)
            result_ph = st.empty()
            alert_ph  = st.empty()

            if st.button("⬡  Run Detection"):
                with st.spinner("Running inference..."):
                    try:
                        data  = predict_api(img_bytes, conf)
                        count = data["count"]
                        dets  = data["detections"]

                        alert_ph.markdown(alert_html(count), unsafe_allow_html=True)
                        img_data = base64.b64decode(data["image"])
                        result_ph.image(img_data, use_container_width=True)

                        if count > 0:
                            st.markdown("<div class='g-section'>Detections</div>",
                                        unsafe_allow_html=True)
                            for i, d in enumerate(dets):
                                st.markdown(
                                    f"<span style='font-family:DM Mono,monospace;"
                                    f"font-size:0.8rem;color:#4A5568;'>"
                                    f"ZONE {i+1}  ·  {d['class_name'].upper()}"
                                    f"  ·  CONF {d['confidence']:.0%}"
                                    f"  ·  BOX {[round(v) for v in d['bbox']]}</span>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")



elif page == "⬡  Video Analysis":
    st.markdown("<div class='g-section'>Video Inference Stream</div>", unsafe_allow_html=True)

    if not api_ok:
        st.error("⚠ Backend API is offline. Start backend.py first.")
        st.stop()

    uploaded = st.file_uploader("", type=["mp4","avi","mov","mkv"],
                                label_visibility="collapsed")

    if uploaded:
        col_v, col_s = st.columns([2, 1])

        with col_s:
            st.markdown("<div class='g-section'>Controls</div>", unsafe_allow_html=True)

            
            start_v = st.button("▶  Start", key="vid_start",
                                use_container_width=True)
            stop_v  = st.button("■  Stop",  key="vid_stop",
                                use_container_width=True)

            st.markdown("<hr class='g-divider'>", unsafe_allow_html=True)
            alert_ph  = st.empty()
            metric_ph = st.empty()
            alert_ph.markdown(idle_alert(), unsafe_allow_html=True)

        with col_v:
            st.markdown("<div class='g-section'>Live Feed</div>", unsafe_allow_html=True)
            frame_ph = st.empty()

       
        if stop_v:
            st.session_state.vid_running   = False
            st.session_state.stop_requested = True

        if start_v:
            st.session_state.vid_running    = True
            st.session_state.stop_requested = False

        if st.session_state.vid_running:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()

            cap          = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx    = 0
            fire_frames  = 0
            total_fires  = 0
            prev_time    = time.time()
            progress     = st.progress(0)

            while cap.isOpened():
              
                if st.session_state.stop_requested:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

               
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * (640 / w))))

               
                try:
                    frame_bytes = encode_frame(frame, quality=70)
                    data        = predict_api(frame_bytes, conf)
                    count       = data["count"]

                    if count > 0:
                        fire_frames += 1
                        total_fires += count

                    
                    img_data = base64.b64decode(data["image"])
                    frame_ph.image(img_data, use_container_width=True)

                except Exception:
                    
                    frame_ph.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True)
                    count = 0

                now       = time.time()
                fps_live  = 1 / (now - prev_time + 1e-8)
                prev_time = now

                alert_ph.markdown(alert_html(count), unsafe_allow_html=True)
                metric_ph.markdown(
                    card("Frame",  f"{frame_idx}/{total_frames}", "c-cyan") +
                    card("Fires",  str(total_fires),               "c-rose") +
                    card("FPS",    f"{fps_live:.1f}",              "c-lime"),
                    unsafe_allow_html=True)

                frame_idx += 1
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

            cap.release()
            os.unlink(tfile.name)
            progress.empty()
            st.session_state.vid_running    = False
            st.session_state.stop_requested = False

            if frame_idx >= total_frames:
                st.success(f"✓ Done — fire in {fire_frames}/{total_frames} frames.")
            else:
                st.info("⬡ Detection stopped.")



elif page == "⬡  Live Camera":
    st.markdown("<div class='g-section'>Live Camera · Backend Inference</div>",
                unsafe_allow_html=True)

    if not api_ok:
        st.error("⚠ Backend API is offline. Start backend.py first.")
        st.stop()

    col_c, col_s = st.columns([2, 1])

    with col_s:
        st.markdown("<div class='g-section'>Controls</div>", unsafe_allow_html=True)

        
        start_c = st.button("▶  Start Camera", key="cam_start",
                            use_container_width=True)
        stop_c  = st.button("■  Stop Camera",  key="cam_stop",
                            use_container_width=True)

        st.markdown("<hr class='g-divider'>", unsafe_allow_html=True)
        alert_ph  = st.empty()
        metric_ph = st.empty()
        alert_ph.markdown(idle_alert(), unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='g-section'>Live Feed</div>", unsafe_allow_html=True)
        frame_ph = st.empty()

   
    if stop_c:
        st.session_state.live_running    = False
        st.session_state.stop_requested  = True

    if start_c:
        st.session_state.live_running    = True
        st.session_state.stop_requested  = False

    if st.session_state.live_running:
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        
        for _ in range(3):
            cap.read()

        if not cap.isOpened():
            st.error("❌ Cannot open camera.")
            st.session_state.live_running = False
        else:
            prev_time      = time.time()
            fire_frames    = 0
            total_frames   = 0
            frame_skip     = 0
            last_img_data  = None   
            count          = 0      

            while st.session_state.live_running:
                if st.session_state.stop_requested:
                    break

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                total_frames += 1
                frame_skip   += 1

                
                if frame_skip % 2 == 0:
                    try:
                        frame_bytes   = encode_frame(frame, quality=65)
                        data          = predict_api(frame_bytes, conf)
                        count         = data["count"]
                        last_img_data = base64.b64decode(data["image"])

                        if count > 0:
                            fire_frames += 1

                    except Exception:
                        
                        last_img_data = None
                        count = 0

                
                if last_img_data is not None:
                    frame_ph.image(last_img_data, use_container_width=True)
                else:
                    frame_ph.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True)

                now       = time.time()
                fps_live  = 1 / (now - prev_time + 1e-8)
                prev_time = now

                alert_ph.markdown(alert_html(count), unsafe_allow_html=True)
                metric_ph.markdown(
                    card("FPS",    f"{fps_live:.1f}",  "c-lime") +
                    card("Fires",  str(fire_frames),   "c-rose") +
                    card("Frames", str(total_frames),  "c-cyan"),
                    unsafe_allow_html=True)

            cap.release()
            frame_ph.empty()
            st.session_state.live_running    = False
            st.session_state.stop_requested  = False
            st.info("⬡ Camera stopped.")

