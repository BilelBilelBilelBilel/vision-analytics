"""
Step 3 — Inference benchmark (PyTorch vs ONNX) + spatial occupancy heatmap.

What it does : runs the same video through (a) the PyTorch model and (b) an ONNX
               export of it, measures FPS / latency, writes output/benchmark.txt.
               During the PyTorch pass it also tracks people and accumulates their
               centroids into a heatmap -> output/heatmap.jpg.
Assignment   : Step 3 (performance optimisation + spatial analytics).
How to run   : python step3_benchmark.py --source data/people.mp4
               (webcam works too, but a file gives a fair, repeatable comparison)

WHY ONNX? ONNX is a framework-neutral model format. ONNX Runtime applies graph
optimisations (operator fusion, constant folding) and is the usual first step before
TensorRT/OpenVINO on edge devices. Whether it beats PyTorch depends on the CPU and
library versions — that is exactly why we MEASURE instead of assuming.
"""
import argparse
import importlib.util
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS = 0
MAX_FRAMES = 300        # cap the benchmark length so it finishes in ~1 minute on CPU
WARMUP_FRAMES = 5       # first calls are slow (memory allocation); not timed
IMG_SIZE = 640


def run_benchmark(model, source, on_detections=None):
    """Run `model` on up to MAX_FRAMES frames. Returns (frames, fps, avg_ms, last_frame)."""
    cap = cv2.VideoCapture(source)
    latencies, frame = [], None
    for i in range(MAX_FRAMES + WARMUP_FRAMES):
        ok, img = cap.read()
        if not ok:
            break
        frame = img
        t0 = time.perf_counter()
        result = model(img, imgsz=IMG_SIZE, classes=[PERSON_CLASS], verbose=False)[0]
        dt = time.perf_counter() - t0          # only the model call is timed
        if i >= WARMUP_FRAMES:
            latencies.append(dt)
        if on_detections:                       # tracking/heatmap work is NOT timed
            on_detections(sv.Detections.from_ultralytics(result))
    cap.release()
    n = len(latencies)
    total = sum(latencies) or 1e-9
    return n, n / total, 1000 * total / max(n, 1), frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file path (default: webcam 0)")
    args = parser.parse_args()
    source = args.source if args.source else 0

    rows = []  # (backend, frames, fps, avg_ms)

    # ---- (a) PyTorch + heatmap accumulation --------------------------------
    print("Benchmarking PyTorch ...")
    tracker = sv.ByteTrack()
    centroids = []

    def collect_centroids(detections):
        tracked = tracker.update_with_detections(detections)
        centroids.extend(tracked.get_anchors_coordinates(sv.Position.CENTER).tolist())

    n, fps, ms, last_frame = run_benchmark(YOLO(MODEL_PATH), source, collect_centroids)
    if last_frame is None:
        raise SystemExit(f"Could not read from source: {source}")
    rows.append(("PyTorch (.pt)", n, fps, ms))

    # ---- (b) ONNX ----------------------------------------------------------
    if importlib.util.find_spec("onnxruntime") is None:
        print("NOTE: onnxruntime is not installed -> skipping the ONNX benchmark.")
    else:
        try:
            print("Exporting to ONNX (first time only, may auto-install the 'onnx' package) ...")
            onnx_path = YOLO(MODEL_PATH).export(format="onnx", imgsz=IMG_SIZE)  # -> yolov8n.onnx
            print("Benchmarking ONNX Runtime ...")
            n, fps, ms, _ = run_benchmark(YOLO(onnx_path, task="detect"), source)
            rows.append(("ONNX Runtime (.onnx)", n, fps, ms))
        except Exception as e:  # keep the prototype alive if export/runtime fails
            print(f"NOTE: ONNX benchmark skipped ({e})")

    # ---- Comparison table --------------------------------------------------
    lines = [f"{'backend':<22}{'total frames':>14}{'FPS':>10}{'avg latency ms':>18}",
             "-" * 64]
    lines += [f"{b:<22}{n:>14}{f:>10.1f}{m:>18.1f}" for b, n, f, m in rows]
    table = "\n".join(lines)
    print("\n" + table)
    with open("output/benchmark.txt", "w") as fh:
        fh.write(table + "\n")

    # ---- Heatmap: 2D histogram of centroids, coloured and blended ----------
    h, w = last_frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for x, y in centroids:                                  # 2D histogram: +1 per visit
        heat[min(int(y), h - 1), min(int(x), w - 1)] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)        # smooth into a density map
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(last_frame, 0.55, colored, 0.45, 0)
    cv2.imwrite("output/heatmap.jpg", overlay)
    print(f"\nSaved output/benchmark.txt and output/heatmap.jpg ({len(centroids)} centroids)")


if __name__ == "__main__":
    main()
