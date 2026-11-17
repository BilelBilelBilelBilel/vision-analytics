# Real-Time Visual Analytics with Ultralytics YOLO and OpenCV

A small, educational proof of concept for a real-time visual analytics
architecture. OpenCV reads a video file or webcam stream. A single-stage YOLO
detector finds people in each frame, and ByteTrack gives each person an ID that
stays the same from frame to frame. Polygon regions of interest (ROIs) turn those
tracks into occupancy counts. The counts are then used three ways: sent as
lightweight JSON telemetry to a FastAPI + SQLite backend, benchmarked across two
inference backends (PyTorch and ONNX Runtime), and checked against a threshold
rule that fires a webhook alert, the kind of request an automation tool such as
n8n receives. Each step is one short script you can read from top to bottom. The
goal is to show the concepts clearly; it is not production code.

## Directory layout

```
vision-analytics/
├── requirements.txt      Python dependencies (CPU only)
├── README.md             This file: overview + concepts
├── RUN.md                Exact step-by-step run instructions
├── ARCHITECTURE.md       How the pieces connect (diagram + roles)
├── data/                 Put your input video here (e.g. data/people.mp4); events.db is created here
│   └── .gitkeep
├── step1_tracking.py     Step 1: YOLO detection + ByteTrack IDs + ROI zone counting → output/annotated.mp4
├── step2_telemetry.py    Step 2: same pipeline emitting JSON events to a FastAPI + SQLite API
├── step3_benchmark.py    Step 3: PyTorch vs ONNX FPS benchmark + spatial occupancy heatmap
├── step4_alerts.py       Step 4: sustained-occupancy rule that fires a webhook alert with a snapshot
└── output/               Generated results: annotated video, heatmap, benchmark table, alert frames
```

## Files at a glance

| File | What it does |
|---|---|
| `step1_tracking.py` | Detects people, tracks them with persistent IDs, counts them in two polygon zones and saves an annotated video. |
| `step2_telemetry.py` | Every 30 frames, POSTs `{timestamp, zone_name, object_count, track_ids}` to `/events`, which stores it in SQLite. `GET /events` returns the last 50 events. |
| `step3_benchmark.py` | Measures FPS and latency for the `.pt` and `.onnx` models, writes `output/benchmark.txt`, and builds `output/heatmap.jpg` from tracked centroids. |
| `step4_alerts.py` | If a zone has at least *N* people for 10 frames in a row, it saves the triggering frame and POSTs an alert to a webhook (n8n-style). If the webhook can't be reached, it prints the alert instead. |

## Key Concepts Demonstrated

| Theory term | Where | What to look at |
|---|---|---|
| **Single-stage detection** | Step 1 (all steps) | `model(frame)` predicts boxes and classes in a single forward pass, with no separate region-proposal stage (unlike two-stage detectors such as Faster R-CNN). This is why YOLO is fast enough for real time. |
| **NMS / IoU** | Step 1 (inside Ultralytics) | YOLO returns many overlapping candidate boxes. Non-Maximum Suppression keeps the highest-confidence box and drops any other box whose IoU (Intersection over Union) with it is above a threshold. ByteTrack also uses IoU to match new boxes to existing tracks. |
| **ByteTrack / Kalman filtering** | Step 1 → `sv.ByteTrack()` | A Kalman filter predicts where each track will be in the next frame. New detections are matched to those predictions by IoU. ByteTrack also uses *low-confidence* boxes in a second matching round, which keeps IDs stable when people are partly hidden. |
| **ROI counting** | Step 1 → `sv.PolygonZone` | A person counts as "in the zone" when the bottom-center of their box (roughly their feet) is inside the polygon. Counting per zone turns pixels into a business metric such as queue length. |
| **Edge-to-cloud telemetry** | Step 2 | The camera loop (edge) does not send video. It sends small JSON events about once per second to an API (cloud), which stores them in a database. |
| **Model optimisation** | Step 3 | The model is exported to ONNX and run with ONNX Runtime, then compared with PyTorch by FPS and latency. |
| **Spatial analytics (heatmap)** | Step 3 | Tracked centroids are collected into a 2D histogram, blurred, colour-mapped and blended over a frame. |
| **Event-driven alerting** | Step 4 | A rule with persistence (≥ threshold for 10 consecutive frames) fires one webhook per episode, which n8n can route to Slack, email and other tools. |

See **RUN.md** to run it and **ARCHITECTURE.md** for the data flow.
