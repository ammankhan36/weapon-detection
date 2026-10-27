"""
Weapon Detector - inference script (YOLOv5 custom model)

Usage examples:
    python detect.py --weights best.pt --source 0                  # webcam
    python detect.py --weights best.pt --source video.mp4          # video file
    python detect.py --weights best.pt --source photo.jpg          # single image
    python detect.py --weights best.pt --source video.mp4 --save   # save annotated output

Press 'q' to quit in video/webcam mode.
"""
import argparse
import sys
import pathlib
import time
from datetime import datetime
from pathlib import Path

# Fix for loading Colab/Linux trained YOLOv5 models on Windows
if sys.platform == "win32":
    try:
        import importlib
        _local = importlib.import_module("pathlib._local")
        _local.PosixPath = _local.WindowsPath
    except Exception:
        pass
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALERT_COOLDOWN = 3  # seconds between alerts/snapshots


def load_model(weights, conf, iou):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, trust_repo=True)
    model.conf = conf  # confidence threshold
    model.iou = iou    # NMS IoU threshold
    model.to(device)
    print(f"[INFO] Model loaded on {device}. Classes: {model.names}")
    return model


def detect_and_draw(model, frame, img_size):
    """Run detection on a BGR frame, draw red boxes, return (frame, detections)."""
    results = model(frame[:, :, ::-1], size=img_size)  # BGR -> RGB
    detections = []
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].tolist():
        label = f"{model.names[int(cls)]} {conf:.2f}"
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 8), (p1[0] + tw + 4, p1[1]), (0, 0, 255), -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        detections.append((model.names[int(cls)], conf))
    if detections:
        cv2.putText(frame, "WARNING: WEAPON DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return frame, detections


def raise_alert(frame, detections, out_dir):
    """Beep (Windows only) and save a timestamped snapshot."""
    try:
        import winsound
        winsound.Beep(1500, 400)
    except Exception:
        print("\a", end="")  # terminal bell fallback
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"alert_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    names = ", ".join(f"{n}({c:.2f})" for n, c in detections)
    print(f"[ALERT] {ts} -> {names} | snapshot: {path}")


def run_image(model, source, args, out_dir):
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    frame, detections = detect_and_draw(model, frame, args.img)
    print(f"[INFO] Detections: {detections if detections else 'none'}")
    out_path = out_dir / f"result_{Path(source).name}"
    cv2.imwrite(str(out_path), frame)
    print(f"[INFO] Annotated image saved: {out_path.resolve()}")
    if not args.noshow:
        try:
            cv2.imshow("Weapon Detector (Press any key to close)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[NOTE] GUI window display skipped ({e}). Image is saved at {out_path}")


def run_stream(model, source, args, out_dir):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera/video source '{source}'.")
        if str(source) == "0":
            print("[HINT] If your computer does not have a connected webcam, test with an image instead:")
            print("       python detect.py --weights best.pt --source OIP.jpg\n")
        return

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(str(out_dir / "output.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_alert, prev_time = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, detections = detect_and_draw(model, frame, args.img)

        if detections and time.time() - last_alert > ALERT_COOLDOWN:
            raise_alert(frame, detections, out_dir)
            last_alert = time.time()

        now = time.time()
        fps_txt = f"FPS: {1 / max(now - prev_time, 1e-6):.1f}"
        prev_time = now
        cv2.putText(frame, fps_txt, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("Weapon Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Weapon detector (YOLOv5)")
    parser.add_argument("--weights", default="best.pt", help="path to trained weights")
    parser.add_argument("--source", default="0", help="0 for webcam, or path to image/video")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img", type=int, default=640, help="inference image size")
    parser.add_argument("--save", action="store_true", help="save annotated video")
    parser.add_argument("--noshow", action="store_true", help="save outputs without opening desktop GUI window")
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    model = load_model(args.weights, args.conf, args.iou)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(model, args.source, args, out_dir)
    else:
        run_stream(model, args.source, args, out_dir)


if __name__ == "__main__":
    main()
