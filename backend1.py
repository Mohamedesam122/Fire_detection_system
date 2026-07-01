import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


app = FastAPI(
    title="YOLOv8 Fire Detection API",
    description="Backend API for real-time fire detection using YOLOv8",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = YOLO("best.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    
    try:
        model = YOLO("yolov8s.pt")
        print("Fallback model yolov8s.pt loaded successfully.")
    except Exception as fallback_err:
        print(f"Error loading fallback model: {fallback_err}")
        model = None

@app.get("/health")
def health_check():
    """Health check endpoint to verify API and model status."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "healthy",
        "model": getattr(model, "model_name", "YOLOv8 Custom"),
        "classes": model.names
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.40, ge=0.0, le=1.0)
):
    """
    Receives an uploaded image, runs YOLO inference, and returns
    detected bounding boxes, confidence scores, and the annotated image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    
    try:
        
        results = model.predict(source=img, conf=conf, save=False)
        result = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    
    detections = []
    boxes = result.boxes
    for box in boxes:
       
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy
        
        score = float(box.conf[0].item())
        
        cls_id = int(box.cls[0].item())
        class_name = model.names.get(cls_id, "unknown")
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": score,
            "class_id": cls_id,
            "class_name": class_name
        })

    
    try:
        annotated_img = result.plot()  
        _, buffer = cv2.imencode('.jpg', annotated_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to annotate image: {e}")

    return {
        "count": len(detections),
        "detections": detections,
        "image": encoded_image
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
