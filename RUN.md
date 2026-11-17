# How to Run (from a fresh machine)

These instructions assume no prior setup. Everything runs on the **CPU**. Nothing
needs to be configured by hand: the YOLO weights (`yolov8n.pt`, about 6 MB) download
automatically the first time you run a script.

## 0. Prerequisites

- **Python 3.10 – 3.13** ([python.org](https://www.python.org/downloads/)). On Windows,
  tick *"Add Python to PATH"* during installation.
- An internet connection for the first install (PyTorch is a large download, a few hundred MB).

Open a terminal **inside the `vision-analytics/` folder**. Every command below is
run from there, because the scripts use relative paths (`data/`, `output/`).

```bash
cd vision-analytics
```

## 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> If PowerShell blocks the script, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> (or use Command Prompt: `.venv\Scripts\activate.bat`)

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the environment is active, your prompt starts with `(.venv)`.

## 2. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Get a sample video

Any video with people walking in it will work. To use the free Intel sample clip:

**Windows (PowerShell)** — type `curl.exe`, because plain `curl` is a PowerShell alias for a different command:
```powershell
curl.exe -L -o data/people.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
```

**Linux / macOS:**
```bash
curl -L -o data/people.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
```

Or copy your own `.mp4` into `data/` and use its name in the commands below.

> **Webcam fallback:** leave out `--source` and the scripts open webcam index `0`.
> A webcam stream never ends, so stop it with **Ctrl+C** (or **q** in the Step 1 preview window).

## 4. Step 1 — Detection, tracking and zone counting

```bash
python step1_tracking.py --source data/people.mp4
```
Add `--show` to watch it live in a window. The script prints the zone counts every 30 frames.
**Result:** `output/annotated.mp4` (boxes, `#ID` labels, both zones and their counts).

## 5. Step 2 — Telemetry API (FastAPI + SQLite)

```bash
python step2_telemetry.py --source data/people.mp4
```
This starts the API at `http://127.0.0.1:8000`, processes the video, and sends one event
per zone every 30 frames. **When the video ends, the API keeps running** so you can query it.
Open a **second terminal** (activate the venv again in it) and run:

**Read the last 50 events:**
```bash
curl http://127.0.0.1:8000/events
```

**Add an event by hand** (Linux/macOS):
```bash
curl -X POST http://127.0.0.1:8000/events -H "Content-Type: application/json" -d '{"timestamp":"2026-01-01T12:00:00Z","zone_name":"queue zone","object_count":3,"track_ids":[1,4,7]}'
```

**Same in Windows PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/events -ContentType "application/json" -Body '{"timestamp":"2026-01-01T12:00:00Z","zone_name":"queue zone","object_count":3,"track_ids":[1,4,7]}'
Invoke-RestMethod http://127.0.0.1:8000/events
```

You can also open **http://127.0.0.1:8000/docs** in a browser and try both endpoints
from the interactive Swagger UI. Press **Ctrl+C** in the first terminal to stop.
**Result:** `data/events.db` (SQLite, table `events`).

## 6. Step 3 — Benchmark (PyTorch vs ONNX) + heatmap

```bash
python step3_benchmark.py --source data/people.mp4
```
The script runs up to 300 frames through each backend. The first run exports
`yolov8n.onnx`; Ultralytics may also install the small `onnx` package for this
automatically. If `onnxruntime` is missing, the ONNX part is skipped with a note.
**Results:** the comparison table is printed to the console and saved to
`output/benchmark.txt`; the heatmap is saved to `output/heatmap.jpg`.

## 7. Step 4 — Threshold alerts to a webhook

```bash
python step4_alerts.py --source data/people.mp4
```
The default webhook is `http://localhost:5678/webhook/test`, the address an n8n
Webhook node with path `test` listens on. **If nothing is listening there, the
alert payload is printed in the console and the script keeps running.** This is
expected and is enough to show how alerting works.

The default threshold is 5 people. If your video rarely has 5 people in one zone,
lower it to see an alert fire:
```bash
python step4_alerts.py --source data/people.mp4 --threshold 2
```

**Send alerts to a different webhook:**
```bash
python step4_alerts.py --source data/people.mp4 --threshold 2 --webhook https://webhook.site/<your-unique-id>
```
(Open https://webhook.site in a browser to get a temporary URL and watch requests arrive.)

**Using a real n8n instance (optional):**
1. Run `npx n8n` (requires Node.js), then open http://localhost:5678.
2. Create a workflow with a **Webhook** node: method `POST`, path `test`.
3. **Activate** the workflow. The production URL `/webhook/test` only works while the
   workflow is active; while you click "Listen for test event", n8n uses `/webhook-test/test` instead:
   ```bash
   python step4_alerts.py --source data/people.mp4 --threshold 2 --webhook http://localhost:5678/webhook-test/test
   ```

**Results:** `output/alert_frame_<timestamp>.jpg` for each alert, plus the JSON payload
`{alert_type, zone, count, timestamp, frame_path}`, either received by the webhook or printed in the console.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not read from source` | Check the video path, or that no other app is using the webcam. |
| `Port 8000 is already in use` (Step 2) | Run with `--port 8001` and use `8001` in the curl URLs. |
| `FutureWarning: ByteTrack was deprecated` | Harmless. `requirements.txt` pins supervision 0.30.x, the last version that still includes `sv.ByteTrack`. |
| Step 4 says `Webhook failed (HTTPStatusError)` | The URL can be reached but returned an error. With n8n, this usually means the workflow isn't active: activate it, or use the `/webhook-test/` URL. |
| Slow (< 5 FPS) | This is normal on an older CPU. Use a shorter or lower-resolution video. |
| `--show` window doesn't open on Linux | Run without `--show`; the output video is still saved. |
