"""
FastAPI Server for Weapon Detection System (YOLOv5)
Provides REST and frame-streaming endpoints for the Web Frontend.
"""
import base64
import io
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix for loading Colab/Linux trained YOLOv5 weights on Windows
if sys.platform == "win32":
    try:
        import importlib
        _local = importlib.import_module("pathlib._local")
        _local.PosixPath = _local.WindowsPath
    except Exception:
        pass
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except Exception as e:
        print(f"[WARNING] Model pre-load deferred or error: {e}")
    yield

app = FastAPI(
    title="Shield AI - Weapon Detection API",
    version="1.0.0",
    description="Real-time weapon detection inference service using YOLOv5",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path(__file__).parent / "best.pt"
SAMPLE_PATH = Path(__file__).parent / "OIP.jpg"

CLASS_COLORS = {
    "Handgun": (0, 0, 255),   # Red
    "Knife": (0, 140, 255),   # Orange
    "Rifle": (0, 50, 255),    # Crimson
    "Sword": (180, 0, 255),   # Purple/Magenta
}

def get_model():
    global MODEL
    if MODEL is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
        print(f"[INFO] Loading YOLOv5 model from {WEIGHTS_PATH} on {DEVICE}...")
        MODEL = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(WEIGHTS_PATH),
            trust_repo=True
        )
        MODEL.to(DEVICE)
        print(f"[INFO] Model ready. Detected classes: {MODEL.names}")
    return MODEL

@app.get("/api/status")
def get_status():
    """Health & model status check."""
    try:
        model = get_model()
        return {
            "status": "online",
            "model_ready": True,
            "device": DEVICE,
            "classes": model.names,
            "weights": WEIGHTS_PATH.name
        }
    except Exception as e:
        return {
            "status": "error",
            "model_ready": False,
            "error": str(e),
            "device": DEVICE
        }

@app.get("/api/sample")
def get_sample_image():
    """Returns sample weapon image for quick frontend testing."""
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    with open(SAMPLE_PATH, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")

def process_frame(model, frame_bgr, conf: float = 0.5, iou: float = 0.45, img_size: int = 640):
    """Core detection helper: runs model, annotates frame, formats JSON output."""
    model.conf = conf
    model.iou = iou

    t0 = time.time()
    # YOLOv5 expects RGB
    results = model(frame_bgr[:, :, ::-1], size=img_size)
    infer_ms = (time.time() - t0) * 1000

    annotated = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    detections = []

    for item in results.xyxy[0].tolist():
        x1, y1, x2, y2, c, cls_idx = item
        cls_name = model.names.get(int(cls_idx), f"Class {int(cls_idx)}")
        conf_pct = round(float(c) * 100, 1)

        bgr_col = CLASS_COLORS.get(cls_name, (0, 0, 255))
        hex_col = f"#{bgr_col[2]:02x}{bgr_col[1]:02x}{bgr_col[0]:02x}"

        # Draw bounding box
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, bgr_col, 2)

        # Draw badge header
        label = f"{cls_name} {conf_pct:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (p1[0], max(0, p1[1] - th - 8)), (p1[0] + tw + 6, p1[1]), bgr_col, -1)
        cv2.putText(
            annotated, label, (p1[0] + 3, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        detections.append({
            "class": cls_name,
            "confidence": round(float(c), 3),
            "percentage": conf_pct,
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "normalized_box": [
                round(x1 / w, 4), round(y1 / h, 4),
                round(x2 / w, 4), round(y2 / h, 4)
            ],
            "color": hex_col
        })

    # Encode annotated frame to JPEG base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return {
        "detections": detections,
        "total_found": len(detections),
        "annotated_image": b64_img,
        "inference_time_ms": round(infer_ms, 1),
        "width": w,
        "height": h,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.45)
):
    """Inference on single uploaded image file."""
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    res = process_frame(model, frame, conf=conf, iou=iou)
    return JSONResponse(content=res)

class Base64FramePayload(BaseModel):
    image: str
    conf: float = 0.5
    iou: float = 0.45

@app.post("/api/detect/frame")
def detect_base64_frame(payload: Base64FramePayload):
    """Inference on base64 snapshot (used for real-time webcam frame-by-frame)."""
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    raw_data = payload.image
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_data)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decoded image is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {e}")

    res = process_frame(model, frame, conf=payload.conf, iou=payload.iou)
    return JSONResponse(content=res)

# Serve Frontend static assets and index.html
if FRONTEND_DIR.exists():
    @app.get("/")
    def serve_frontend_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Weapon detection API is online. Frontend index not found."}

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Shield AI Weapon Detector Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    print(f"[INFO] Starting Shield AI Weapon Detector Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
