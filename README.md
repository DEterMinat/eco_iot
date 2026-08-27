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
├── models/                  # AI model weights (TFLite / ONNX)
│   └── waste_classifier.tflite
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
| TFLite Model    | ~50 MB     |
| OpenCV + Camera | ~80 MB     |
| FastAPI Server  | ~40 MB     |
| Frame Buffer    | ~30 MB     |
| OS Reserved     | ~200 MB    |
| **Total**       | **~460 MB** (well within 1GB) |
