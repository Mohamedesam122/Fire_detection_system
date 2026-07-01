# 🔥 Fire Detection System

An end-to-end AI-powered fire detection system built using YOLOv8 and FastAPI. The project covers the complete machine learning lifecycle, from dataset preparation and model training to deployment and API integration for real-time fire detection.
<img width="1920" height="935" alt="image" src="https://github.com/user-attachments/assets/a445b211-785f-463d-9218-7b990422d7b2" />
<img width="1920" height="894" alt="image" src="https://github.com/user-attachments/assets/b104b4ca-b1a1-4db8-9ccc-b91c77e51d35" />


## 🚀 Overview

This project uses a custom fire dataset to train a YOLOv8 object detection model capable of identifying fire regions in images. The trained model is exposed through a FastAPI REST API, making it easy to integrate into web, mobile, or monitoring applications.

The system supports:

- Custom YOLOv8 training
- Model evaluation and performance analysis
- Real-time image inference
- REST API integration using FastAPI
- ONNX model export
- Production deployment

---

## 🛠️ Tech Stack

### AI & Computer Vision
- YOLOv8
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

### Backend
- FastAPI
- Uvicorn

### Dataset Management
- Roboflow

### Deployment
- Docker (Optional)
- ONNX Runtime (Optional)

---

## 📂 Dataset

The model was trained on a custom fire detection dataset exported in YOLOv8 format.

Example dataset configuration:

```yaml
train: train/images
val: valid/images

nc: 1

names:
  - fire
```

Dataset structure:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

---

## 🏋️ Model Training

Training was performed using YOLOv8 with data augmentation and cosine learning rate scheduling.

Example training configuration:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",
    imgsz=640,
    batch=32,
    epochs=100,
    patience=20,
    optimizer="SGD",
    cos_lr=True,
    lr0=0.01,
    mosaic=1.0,
    copy_paste=0.1,
    degrees=3.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)
```

---

## 📊 Evaluation Metrics

The model is evaluated using standard object detection metrics:

### mAP@50
Measures detection quality when IoU ≥ 0.50.

### mAP@50:95
A stricter metric that averages performance across IoU thresholds from 0.50 to 0.95.

### Precision
Measures how many predicted fire detections are actually correct.

### Recall
Measures how many real fire instances were successfully detected.

### F1 Score
Harmonic mean of Precision and Recall.

Example validation results:

```text
mAP@50      = 0.860
mAP@50:95   = 0.634
Precision   = 0.783
Recall      = 0.789
F1 Score    = 0.786
```

---

## 📈 Training Analysis

Training performance is monitored using:

- mAP curves
- Precision & Recall curves
- F1 Score curves
- Box Loss
- Classification Loss
- DFL Loss
- Precision-Recall Curve
- Confusion Matrix

These metrics help evaluate convergence and model stability during training.

---

## 🧠 Understanding the Loss Functions

### Box Loss
Measures how accurately the predicted bounding box matches the ground truth.

### Classification Loss (Cls Loss)
Measures classification accuracy of detected objects.

### DFL Loss (Distribution Focal Loss)
Measures the precision of bounding box edge localization.

Lower values indicate better performance.

---

## 🔍 Running Inference

### Image Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="fire.jpg",
    conf=0.25,
    save=True
)
```

### Access Detection Results

```python
for result in results:
    for box in result.boxes:
        print("Confidence:", float(box.conf))
        print("Coordinates:", box.xyxy.tolist())
```

---

## 🌐 FastAPI Integration

### Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API will be available at:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### Predict Fire

```http
POST /predict
```

Upload an image and receive fire detection results.

Request:

```form-data
file=image.jpg
```

Example Response:

```json
{
  "fire_detected": true,
  "confidence": 0.94,
  "detections": [
    {
      "class": "fire",
      "confidence": 0.94,
      "bbox": {
        "x1": 120,
        "y1": 85,
        "x2": 320,
        "y2": 295
      }
    }
  ]
}
```

---

## 📦 Export Model

Export the trained model to ONNX format:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

onnx_path = model.export(
    format="onnx",
    imgsz=640,
    simplify=True
)
```

---

## 🖥️ Real-Time Webcam Detection

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.25)

    annotated_frame = results[0].plot()

    cv2.imshow("Fire Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📁 Project Structure

```text
fire-detection-system/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── utils.py
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── best.pt
│   └── best.onnx
│
├── notebooks/
│
├── runs/
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

---

## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fire-detection-system.git
cd fire-detection-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

```text
ultralytics
torch
opencv-python
fastapi
uvicorn
numpy
pandas
matplotlib
roboflow
```

---

## 🎯 Future Improvements

- Video stream processing
- RTSP camera support
- Fire and smoke multi-class detection
- Edge deployment
- Docker containerization
- Cloud deployment
- Real-time alerting system

---

## 👨‍💻 Author

**Mohamed Esam**

AI Engineer | Computer Vision Engineer

---

## 📜 License

This project is released under the MIT License.
## mo
