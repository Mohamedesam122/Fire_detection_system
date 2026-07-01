"""
FireGuard AI – FastAPI Backend
==============================
Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

Swagger docs: http://localhost:8000/docs
"""

import io
import logging
import os
import smtplib
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fireguard")

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "best.pt")
CONF_THRESH  = float(os.getenv("CONF_THRESH", "0.25"))
DB_PATH      = os.getenv("DB_PATH",      "fireguard.db")

# Email config (set via environment variables)
EMAIL_SENDER   = os.getenv("EMAIL_SENDER",   "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST      = os.getenv("SMTP_HOST",      "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT",  "587"))

# ─── Global model ───────────────────────────────────────────────────────────
model: Optional[YOLO] = None


# ─── Database ───────────────────────────────────────────────────────────────
def init_db():
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
    logger.info("Database initialised at %s", DB_PATH)


def log_detection(source_type: str, fire_count: int,
                  max_conf: float, avg_conf: float, fps: float = None):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO detections (timestamp, source_type, fire_count, max_conf, avg_conf, fps) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), source_type, fire_count,
         round(max_conf, 4), round(avg_conf, 4), fps),
    )
    con.commit()
    con.close()


# ─── Email alert ────────────────────────────────────────────────────────────
def send_email_alert(fire_count: int, max_conf: float, source: str):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        logger.warning("Email credentials not configured – skipping alert.")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🔥 FireGuard AI – Fire Detected!"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER

        html = f"""
        <html><body style="font-family:Arial;background:#1a1a2e;color:#eee;padding:20px">
          <h2 style="color:#ff4444">🔥 Fire Alert!</h2>
          <p><b>Source:</b> {source}</p>
          <p><b>Fires detected:</b> {fire_count}</p>
          <p><b>Max confidence:</b> {max_conf:.1%}</p>
          <p><b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
          <hr style="border-color:#444">
          <p style="color:#aaa;font-size:12px">FireGuard AI – Automated Alert</p>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.starttls()
            srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logger.info("Email alert sent to %s", EMAIL_RECEIVER)
    except Exception as e:
        logger.error("Failed to send email: %s", e)


# ─── Pydantic schemas ────────────────────────────────────────────────────────
class Detection(BaseModel):
    cls:        str
    confidence: float
    bbox:       List[float]   # [x1, y1, x2, y2]


class PredictResponse(BaseModel):
    status:          str
    source_type:     str
    fire_count:      int
    max_confidence:  float
    avg_confidence:  float
    processing_ms:   float
    alert_sent:      bool
    detections:      List[Detection]


class HistoryItem(BaseModel):
    id:          int
    timestamp:   str
    source_type: str
    fire_count:  int
    max_conf:    float
    avg_conf:    float
    fps:         Optional[float]


class StatsResponse(BaseModel):
    total_detections: int
    total_fires:      int
    avg_confidence:   float
    most_recent:      Optional[str]


# ─── Lifespan (load model + db once) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading YOLOv8 model from %s …", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded ✅")
    init_db()
    yield
    logger.info("Shutting down FireGuard AI backend.")


# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "FireGuard AI",
    description = "Real-Time Fire Detection API powered by YOLOv8",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Helpers ────────────────────────────────────────────────────────────────
def run_inference(frame: np.ndarray, source_type: str,
                  send_alert: bool = True) -> PredictResponse:
    t0 = time.perf_counter()
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections: List[Detection] = []
    for r in results:
        for box in r.boxes:
            detections.append(Detection(
                cls        = "fire",
                confidence = round(float(box.conf[0]), 4),
                bbox       = [round(v, 2) for v in box.xyxy[0].tolist()],
            ))

    fire_count = len(detections)
    confs      = [d.confidence for d in detections]
    max_conf   = max(confs) if confs else 0.0
    avg_conf   = round(sum(confs) / len(confs), 4) if confs else 0.0

    alert_sent = False
    if fire_count > 0:
        log_detection(source_type, fire_count, max_conf, avg_conf)
        if send_alert:
            send_email_alert(fire_count, max_conf, source_type)
            alert_sent = True

    return PredictResponse(
        status         = "success",
        source_type    = source_type,
        fire_count     = fire_count,
        max_confidence = max_conf,
        avg_confidence = avg_conf,
        processing_ms  = round(elapsed_ms, 2),
        alert_sent     = alert_sent,
        detections     = detections,
    )


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "name":        "FireGuard AI",
        "version":     "1.0.0",
        "description": "Real-Time Fire Detection API",
        "endpoints": {
            "image":   "POST /predict/image",
            "video":   "POST /predict/video",
            "health":  "GET  /health",
            "history": "GET  /history",
            "stats":   "GET  /stats",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status":     "healthy",
        "model":      MODEL_PATH,
        "model_ready": model is not None,
        "timestamp":  datetime.utcnow().isoformat(),
    }


@app.post("/predict/image", response_model=PredictResponse, tags=["Detection"])
async def predict_image(file: UploadFile = File(...)):
    """Upload an image and get fire detections."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        data  = await file.read()
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Could not decode image.")
        return run_inference(frame, source_type="image")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image inference error: %s", e)
        raise HTTPException(500, f"Inference failed: {e}")


@app.post("/predict/video", tags=["Detection"])
async def predict_video(file: UploadFile = File(...)):
    """Upload a video and get per-frame fire detections."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap       = cv2.VideoCapture(tmp_path)
        fps_in    = cap.get(cv2.CAP_PROP_FPS) or 25
        results   = []
        frame_idx = 0
        t0        = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Process every 5th frame for speed
            if frame_idx % 5 == 0:
                res = run_inference(frame, source_type="video", send_alert=False)
                results.append({
                    "frame":       frame_idx,
                    "fire_count":  res.fire_count,
                    "max_conf":    res.max_confidence,
                    "detections":  [d.dict() for d in res.detections],
                })
            frame_idx += 1

        cap.release()
        os.unlink(tmp_path)

        elapsed  = time.perf_counter() - t0
        total_fires = sum(r["fire_count"] for r in results)
        frames_with_fire = sum(1 for r in results if r["fire_count"] > 0)

        if total_fires > 0:
            max_c = max(r["max_conf"] for r in results if r["fire_count"] > 0)
            log_detection("video", total_fires, max_c, 0.0, round(frame_idx / elapsed, 1))
            send_email_alert(total_fires, max_c, "video upload")

        return JSONResponse({
            "status":            "success",
            "source_type":       "video",
            "total_frames":      frame_idx,
            "processed_frames":  len(results),
            "frames_with_fire":  frames_with_fire,
            "total_fire_detections": total_fires,
            "processing_fps":    round(frame_idx / elapsed, 1),
            "frame_results":     results[:50],   # cap response size
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video inference error: %s", e)
        raise HTTPException(500, f"Video processing failed: {e}")


@app.get("/history", response_model=List[HistoryItem], tags=["Analytics"])
async def get_history(limit: int = 50):
    """Return the last N detection events."""
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, timestamp, source_type, fire_count, max_conf, avg_conf, fps "
        "FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [HistoryItem(id=r[0], timestamp=r[1], source_type=r[2],
                        fire_count=r[3], max_conf=r[4],
                        avg_conf=r[5], fps=r[6]) for r in rows]


@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats():
    """Aggregate detection statistics."""
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT COUNT(*), SUM(fire_count), AVG(max_conf), MAX(timestamp) "
        "FROM detections"
    ).fetchone()
    con.close()
    return StatsResponse(
        total_detections = row[0] or 0,
        total_fires      = row[1] or 0,
        avg_confidence   = round(row[2] or 0, 4),
        most_recent      = row[3],
    )


@app.delete("/history", tags=["Analytics"])
async def clear_history():
    """Clear all detection history."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM detections")
    con.commit()
    con.close()
    return {"status": "success", "message": "History cleared."}


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
