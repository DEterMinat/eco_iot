# ECO-Gradian IoT Edge Device Package

> Production-ready AI Edge inference engine for Orange Pi / Raspberry Pi / SBCs.
> Designed for **≤1GB RAM** environments with auto cache/log cleanup, USB camera support,
> and API Key authentication gateway.

## 🏗️ Architecture

```
eco_iot/
├── main.py                  # Entry point & FastAPI edge server
├── config.py                # Configuration & RAM limits
├── ai_engine.py             # Lightweight AI inference (TFLite / ONNX / OpenCV DNN)
├── camera.py                # USB Camera manager (V4L2 / OpenCV)
├── key_manager.py           # API Key generation & authentication
├── memory_guard.py          # RAM watchdog & auto cache/log cleanup
├── models/                  # Frozen YOLOv8 119-class ONNX profiles
│   ├── waste_detector.onnx  # 320x320 CPU profile used by demo
│   └── MODEL_CARD.md        # SHA-256, license and benchmark notes
├── logs/                    # Auto-rotated logs
├── data/                    # Inference cache & telemetry
├── requirements.txt         # Minimal Python dependencies
├── install.sh               # One-shot install script for Orange Pi
├── eco_iot.service           # systemd unit file for auto-start
└── README.md
```

## 🚀 Quick Start

```bash
# On Orange Pi / Raspberry Pi:
cd /opt/eco_iot
pip install -r requirements.txt
python main.py --camera 0 --port 8080
```

The production demo path is OpenCV DNN + `models/waste_detector.onnx` (320x320)
with a single inference worker and a latest-frame queue. If OpenCV or the camera
is unavailable the service reports `503`/`unknown`; it never creates a synthetic
frame or guesses a waste type.

### Orange Pi deployment (offline-safe)

Run these commands on the Orange Pi after copying this directory (no GitHub/DNS
access is required):

```bash
cd ~/eco_iot
python3 -m venv .venv
. .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
export ECO_IOT_PORT=8080
export ECO_IOT_CAMERA=0
export ECO_IOT_CONFIDENCE=0.35
python main.py --camera "$ECO_IOT_CAMERA" --port "$ECO_IOT_PORT"
```

Generate the key once and place it only in the laptop backend `.env` as
`EDGE_API_KEY`; do not put it in Flutter source or browser URLs:

```bash
. .venv/bin/activate
python main.py --generate-key "eco-demo-camera"
```

The laptop backend should point `EDGE_URL` at `http://<orange-pi-ip>:8080` and
the Flutter build should use `--dart-define=API_URL=http://<laptop-ip>:8000`.

## 🔑 API Key System

```bash
# Generate a new API key
python main.py --generate-key "Lab Camera #3"

# Connect from ECO-Gradian mobile app or backend using the key
curl -H "x-api-key: ek_xxxxxxxx" http://192.168.1.100:8080/predict
```

## 🧠 RAM Budget (≤1GB Target)

| Component       | RAM Budget |
|:----------------|:-----------|
| Python Runtime  | ~60 MB     |
| YOLOv8 ONNX 320 | ~14 MB     |
| OpenCV + Camera | ~80 MB     |
| FastAPI Server  | ~40 MB     |
| Frame Buffer    | ~30 MB     |
| OS Reserved     | ~200 MB    |
| **Total**       | **~460 MB** (well within 1GB) |
