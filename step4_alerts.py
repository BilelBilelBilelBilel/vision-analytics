"""
Step 4 — Rule-based alerts + webhook (simulating an n8n automation).

What it does : runs the tracking pipeline; if a zone holds >= THRESHOLD people for
               SUSTAIN_FRAMES consecutive frames, it fires ONE alert: saves the
               triggering frame and POSTs a JSON payload to a webhook URL.
Assignment   : Step 4 (alerting / workflow automation).
How to run   : python step4_alerts.py --source data/people.mp4
               python step4_alerts.py --source data/people.mp4 --threshold 2 \
                      --webhook http://localhost:5678/webhook/test

HOW THIS MAPS TO n8n
- In n8n, a "Webhook" node (path /webhook/test) receives this exact JSON payload.
- Following nodes act on it: e.g. an "IF" node checks `count`, then a Slack/Email/
  Telegram node notifies staff with `zone`, `timestamp` and the frame image path.
- The vision code stays simple: it only says "something happened"; n8n decides who
  gets told and how — business logic can change without touching the CV pipeline.
"""
import argparse
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS = 0
SUSTAIN_FRAMES = 10     # occupancy must stay high this many consecutive frames
ZONES = {
    "queue zone":    [(0.05, 0.40), (0.45, 0.40), (0.45, 0.95), (0.05, 0.95)],
    "entrance zone": [(0.55, 0.20), (0.95, 0.20), (0.95, 0.70), (0.55, 0.70)],
}


def send_alert(url, payload):
    """POST the alert; if the webhook is down, print it instead and keep going."""
    try:
        r = httpx.post(url, json=payload, timeout=3)
        r.raise_for_status()               # treat 4xx/5xx (e.g. inactive n8n workflow) as failure
        print(f"ALERT sent to webhook (HTTP {r.status_code}): {payload}")
    except httpx.HTTPError as e:           # connection refused, timeout, or bad status
        print(f"Webhook failed ({type(e).__name__}) -> fallback, payload was:\n  {payload}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file path (default: webcam 0)")
    parser.add_argument("--webhook", default="http://localhost:5678/webhook/test")
    parser.add_argument("--threshold", type=int, default=5, help="people per zone to alert on")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source if args.source else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source: {args.source or 'webcam 0'}")
    h, w = frame.shape[:2]

    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    zones = {name: sv.PolygonZone(polygon=(np.array(pts) * [w, h]).astype(int))
             for name, pts in ZONES.items()}
    zone_annotators = {name: sv.PolygonZoneAnnotator(zone=z, color=sv.Color.from_hex("#ff4d4d"))
                       for name, z in zones.items()}
    box_annotator, label_annotator = sv.BoxAnnotator(), sv.LabelAnnotator()

    # Per-zone state: how many consecutive frames above threshold, and whether we
    # already alerted for this "episode" (re-armed once occupancy drops again).
    streak = {name: 0 for name in zones}
    alerted = {name: False for name in zones}

    frame_idx = 0
    try:
        while ok:
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))

            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(
                annotated, detections, labels=[f"#{t}" for t in detections.tracker_id])

            counts = {}
            for name, zone in zones.items():
                counts[name] = int(zone.trigger(detections).sum())
                annotated = zone_annotators[name].annotate(annotated)

            # --- The threshold rule ---
            for name, count in counts.items():
                if count >= args.threshold:
                    streak[name] += 1
                else:
                    streak[name], alerted[name] = 0, False     # re-arm
                if streak[name] >= SUSTAIN_FRAMES and not alerted[name]:
                    alerted[name] = True                        # fire ONE alert
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    frame_path = f"output/alert_frame_{ts}.jpg"
                    cv2.imwrite(frame_path, annotated)          # the exact triggering frame
                    send_alert(args.webhook, {
                        "alert_type": "zone_occupancy",
                        "zone": name,
                        "count": count,
                        "timestamp": ts,
                        "frame_path": frame_path,
                    })

            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}: {counts}")
            frame_idx += 1
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
    print(f"Done. {frame_idx} frames processed.")


if __name__ == "__main__":
    main()
