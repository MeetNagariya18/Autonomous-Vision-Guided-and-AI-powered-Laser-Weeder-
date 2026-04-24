# 🌾 Weed Rover — Command Center

A farm-themed web interface for your Laser Weeding Robot detection system.

## Quick Start

### 1. Place your model
Copy `MY_WEED_ROVER_BRAIN.pt` (your trained YOLOv8 weights) into this folder.
If your model has a different name, you can change it in the browser interface.

### 2. Install dependencies
```bash
pip install flask ultralytics opencv-python numpy werkzeug
```

### 3. Run the app

**Windows:** Double-click `run.bat`

**Any platform:**
```bash
python app.py
```

### 4. Open browser
Go to: **http://127.0.0.1:5000**

---

## Features
- 🖼️ **Image mode** — Upload single image, get annotated result + CSV
- 🎬 **Video mode** — Upload video, get fully annotated output video + CSV
- 📊 **Live stats** — Weed count, crop count, pan/tilt angles, target coordinates, confidence
- 💾 **Auto-save** — All outputs saved to `outputs/` folder automatically
- ⬇️ **Download** — One-click download of annotated image/video and CSV

## Output Files
All outputs are saved to the `outputs/` subfolder:
- `<name>_result.jpg` / `<name>_result.mp4` — annotated output
- `<name>_results_<timestamp>.csv` — detection log

## Configuration
You can change detection settings directly in `app.py`:
- `CONF_THRESHOLD` — detection confidence (default 0.55)
- `MAX_WEED_AREA_FRACTION` — max weed box size as % of frame (default 15%)
- `SIZE_RATIO_THRESHOLD` — weed/crop size ratio for reclassification (default 2.5×)
