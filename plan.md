# Shield AI - Weapon Detection Execution Guide

## 1. Backend Web Server (`server.py`)
Runs the FastAPI inference engine and serves the web frontend:

```powershell
# Navigate to backend
cd "c:\Users\amman\Desktop\weapon detection\backend"

# Run the server (default port 8000)
python server.py

# Optional: Run on an alternate port if 8000 is busy
python server.py --port 8080
```
- Open in browser: **http://127.0.0.1:8000** (or **http://localhost:8000**)

---

## 2. Standalone Frontend (`frontend/`)
If you want to run the frontend independently using Python's static server:

```powershell
cd "c:\Users\amman\Desktop\weapon detection\frontend"
python -m http.server 3000
```
- Open in browser: **http://localhost:3000**

---

## 3. CLI Detection Script (`detect.py`)
Run inference directly from the command line without opening the web browser:

### A. Detect on a Single Image
```powershell
cd "c:\Users\amman\Desktop\weapon detection\backend"

# Detect on test image (shows window + saves to output/)
python detect.py --weights best.pt --source OIP.jpg

# Save output without popping up the desktop GUI window
python detect.py --weights best.pt --source OIP.jpg --noshow
```

### B. Detect on a Video File
```powershell
cd "c:\Users\amman\Desktop\weapon detection\backend"
python detect.py --weights best.pt --source path\to\video.mp4 --save
```

### C. Detect on Live Physical Webcam
*(Note: Requires a connected USB or integrated webcam)*
```powershell
cd "c:\Users\amman\Desktop\weapon detection\backend"
python detect.py --weights best.pt --source 0
```

### D. Advanced Confidence & Threshold Parameters
```powershell
# Custom confidence threshold (e.g. 0.4 for higher sensitivity, 0.7 for higher precision)
python detect.py --weights best.pt --source OIP.jpg --conf 0.4 --iou 0.45
```
