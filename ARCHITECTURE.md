# Architecture

```
  EDGE (camera side)
  +--------------------------+
  | Video file / webcam/RTSP |
  +------------+-------------+
               | frames
               v
  +--------------------------+
  | OpenCV VideoCapture      |
  +------------+-------------+
               | BGR numpy arrays
               v
  +--------------------------+       Step 1: output/annotated.mp4
  | YOLO detect (+NMS/IoU)   |       Step 3: FPS benchmark (.pt vs .onnx)
  | ByteTrack  -> track IDs  |-----> Step 3: centroid heatmap
  | PolygonZone -> counts    |
  +-----+--------------+-----+
        |              |
        | Step 2       | Step 4
        | JSON event   | rule: count >= T for 10 frames
        | every N      | -> save output/alert_frame_<ts>.jpg
        | frames       |
  ======|==============|=========  network boundary  ==========
        | HTTP POST    | HTTP POST
        v              v
  +-------------+   +-------------------------+
  | FastAPI     |   | Webhook (n8n)           |
  | /events     |   | -> Slack / e-mail / ... |
  +------+------+   +-------------------------+
         |
         v
  +-------------+
  | SQLite      |
  | events table|
  +-------------+
  CLOUD / SERVICES
```

## Components

**Video source + OpenCV**
- `cv2.VideoCapture` hides where frames come from: a file, a webcam (index `0`) or an RTSP URL.
- It returns frames as NumPy arrays (BGR), which every later stage uses.
- It also gives the frame size, which is used to scale the zone polygons.

**YOLO detector (Ultralytics)**
- Single-stage CNN: one forward pass gives boxes, classes and confidences.
- NMS (based on IoU) removes duplicate boxes. It runs inside `model(frame)`.
- `yolov8n` is the "nano" model: the smallest and fastest, so it runs on a CPU.
- In these scripts it only detects COCO class 0 (`person`).

**ByteTrack tracker (supervision)**
- Turns per-frame detections into tracks that keep the same `tracker_id`.
- A Kalman filter predicts motion, and IoU matching links new boxes to existing tracks.
- Stable IDs mean we count people, not detections (no double counting from frame to frame).

**ROI zones (supervision `PolygonZone`)**
- Two hardcoded polygons, `queue zone` and `entrance zone`, defined as fractions of the frame.
- `zone.trigger(detections)` gives a True/False mask showing which tracks are inside.
- The sum of that mask is the occupancy count, the main metric of the system.

**Telemetry API (FastAPI + SQLite)**
- The edge sends small JSON events instead of video, which saves bandwidth and avoids sending images of people.
- `POST /events` inserts a row (`id, timestamp, zone, count`), and `GET /events` returns the last 50.
- SQLite is a single file (`data/events.db`) with no server to run. A production system would use a server database instead.

**Alerting (webhook → n8n)**
- A rule with persistence (count ≥ T for 10 consecutive frames) means short glitches don't trigger alerts.
- Only one alert is sent per episode. The rule re-arms when occupancy drops below T.
- The payload includes the path of the saved frame that triggered the alert, as evidence.
- If the webhook can't be reached, the alert is printed instead, so the vision loop keeps running.

**Benchmark + heatmap (offline analysis)**
- The same frames are run through PyTorch and ONNX Runtime. Only the model call is timed.
- Heatmap: a 2D histogram of tracked centroids → Gaussian blur → `COLORMAP_JET`, blended over a frame.
