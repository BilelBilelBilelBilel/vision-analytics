"""
Step 2 — Edge-to-cloud telemetry (vision loop -> JSON events -> FastAPI -> SQLite).

What it does : runs the Step 1 pipeline and, every N frames, sends one small JSON
               event per zone to a FastAPI server, which stores it in SQLite.
               After the video ends, the API stays up so you can query it.
Assignment   : Step 2 (telemetry / API / persistence).
How to run   : python step2_telemetry.py --source data/people.mp4
               then:  curl http://127.0.0.1:8000/events
               (Ctrl+C to stop the API)

DESIGN CHOICE: the API runs in a background thread of the SAME process, and the
vision loop calls it over HTTP with httpx. Writing to SQLite directly would be ~2
lines shorter, but going through POST /events mirrors a real deployment, where the
camera ("edge") and the server ("cloud") are different machines. Only the URL changes.
"""
import argparse
import sqlite3
import threading
import time
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
import supervision as sv
import uvicorn
from fastapi import FastAPI
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS = 0
DB_PATH = "data/events.db"
EMIT_EVERY_N_FRAMES = 30          # ~1 event per second at 30 fps: lightweight
ZONES = {
    "queue zone":    [(0.05, 0.40), (0.45, 0.40), (0.45, 0.95), (0.05, 0.95)],
    "entrance zone": [(0.55, 0.20), (0.95, 0.20), (0.95, 0.70), (0.55, 0.70)],
}

# ----------------------------------------------------------------------------
# Part A — the "cloud" side: FastAPI + SQLite
# ----------------------------------------------------------------------------
app = FastAPI(title="Vision telemetry")


def db():
    # A new connection per request keeps things thread-safe and simple.
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS events ("
                 "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, zone TEXT, count INTEGER)")
    return conn


@app.post("/events")
def add_event(event: dict):
    # Expected body: {timestamp, zone_name, object_count, track_ids}
    with db() as conn:
        conn.execute("INSERT INTO events (timestamp, zone, count) VALUES (?, ?, ?)",
                     (event["timestamp"], event["zone_name"], event["object_count"]))
    return {"status": "stored"}


@app.get("/events")
def last_events():
    with db() as conn:
        rows = conn.execute("SELECT id, timestamp, zone, count FROM events "
                            "ORDER BY id DESC LIMIT 50").fetchall()
    return [{"id": r[0], "timestamp": r[1], "zone": r[2], "count": r[3]} for r in rows]


def start_api_in_background(port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:      # wait until the server is ready to accept requests
        if not thread.is_alive():  # uvicorn exits immediately if the port is taken
            raise SystemExit(f"Port {port} is already in use. Try: --port 8001")
        time.sleep(0.1)
    print(f"API running at http://127.0.0.1:{port}  (docs: http://127.0.0.1:{port}/docs)")


# ----------------------------------------------------------------------------
# Part B — the "edge" side: the Step 1 vision loop + event emission
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file path (default: webcam 0)")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()
    api_url = f"http://127.0.0.1:{args.port}/events"

    start_api_in_background(args.port)

    cap = cv2.VideoCapture(args.source if args.source else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source: {args.source or 'webcam 0'}")
    h, w = frame.shape[:2]

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    zones = {name: sv.PolygonZone(polygon=(np.array(pts) * [w, h]).astype(int))
             for name, pts in ZONES.items()}

    frame_idx, sent = 0, 0
    try:
        while ok:
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))

            if frame_idx % EMIT_EVERY_N_FRAMES == 0:
                for name, zone in zones.items():
                    inside = zone.trigger(detections)          # boolean mask
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "zone_name": name,
                        "object_count": int(inside.sum()),
                        "track_ids": [int(t) for t in detections.tracker_id[inside]],
                    }
                    httpx.post(api_url, json=event, timeout=5)  # edge -> cloud
                    sent += 1
                    print("event:", event)

            frame_idx += 1
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Video loop stopped by user.")
    finally:
        cap.release()

    print(f"\nVideo done: {frame_idx} frames, {sent} events sent.")
    print(f"API still running. Try:  curl {api_url}   (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Bye.")


if __name__ == "__main__":
    main()
