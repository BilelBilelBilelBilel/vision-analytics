"""
Step 1 — Detection + Tracking + ROI counting.

What it does : reads a video (or webcam), detects people with YOLO, gives each
               person a persistent ID with ByteTrack, and counts how many people
               are inside two hardcoded zones. Saves output/annotated.mp4.
Assignment   : Step 1 (core vision pipeline).
How to run   : python step1_tracking.py --source data/people.mp4
               python step1_tracking.py              (webcam index 0)
               add --show to open a live preview window (press q to quit)

THEORY IN 4 LINES
- Single-stage detection: YOLO predicts boxes + classes in ONE forward pass over a
  grid (no separate "region proposal" stage), then NMS/IoU removes duplicate boxes.
- ByteTrack: a Kalman filter predicts where each track moves next; new boxes are
  matched to tracks by IoU (high- AND low-confidence boxes) -> stable IDs across frames.
- ROI counting: a polygon zone checks whether each box's bottom-center point lies inside it.
"""
import argparse

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# --- Hardcoded settings (fine for a prototype) ---
MODEL_PATH = "yolov8n.pt"       # downloaded automatically on first run (~6 MB)
PERSON_CLASS = 0                # COCO class 0 = "person"; we only count people
OUTPUT_PATH = "output/annotated.mp4"

# Zones are given as FRACTIONS of the frame (x, y in 0..1) so they work for any
# video resolution. They are converted to pixels once the first frame is read.
ZONES = {
    "queue zone":    [(0.05, 0.40), (0.45, 0.40), (0.45, 0.95), (0.05, 0.95)],
    "entrance zone": [(0.55, 0.20), (0.95, 0.20), (0.95, 0.70), (0.55, 0.70)],
}
ZONE_COLORS = [sv.Color.from_hex("#ff4d4d"), sv.Color.from_hex("#33cc66")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file path (default: webcam 0)")
    parser.add_argument("--show", action="store_true", help="show a live preview window")
    args = parser.parse_args()

    # 1) OpenCV video capture: a file path, or webcam index 0
    cap = cv2.VideoCapture(args.source if args.source else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source: {args.source or 'webcam 0'}")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # webcams sometimes report 0

    # 2) Model, tracker, zones, annotators
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    zones = {
        name: sv.PolygonZone(polygon=(np.array(pts) * [w, h]).astype(int))
        for name, pts in ZONES.items()
    }
    zone_annotators = [
        sv.PolygonZoneAnnotator(zone=z, color=c, thickness=2, display_in_zone_count=False)
        for z, c in zip(zones.values(), ZONE_COLORS)
    ]
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx = 0
    try:
        while ok:
            # 3) Detect (single forward pass; NMS is applied inside Ultralytics)
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # 4) Track: assign persistent IDs (detections.tracker_id)
            detections = tracker.update_with_detections(detections)

            # 5) Count per zone: trigger() returns a True/False mask per detection
            counts = {name: int(z.trigger(detections).sum()) for name, z in zones.items()}

            # 6) Draw everything
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            for (name, z), za in zip(zones.items(), zone_annotators):
                annotated = za.annotate(annotated)
                x, y = z.polygon[0]
                cv2.putText(annotated, f"{name}: {counts[name]}", (int(x) + 5, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)
            if args.show:
                cv2.imshow("Step 1 - tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}: {counts}")
            frame_idx += 1
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Done. {frame_idx} frames -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
