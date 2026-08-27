# ECO-Gradian edge model

- Artifact: `waste_detector.onnx` (source `waste_detector.pt`)
- Task: YOLOv8 detection, 119 classes, input 320x320, opset 12
- Runtime: OpenCV DNN CPU (`opencv_onnx`); RKNN is optional and must pass parity gates
- License: AGPL-3.0 (Ultralytics); deploy only with compatible application licensing
- Demo labels: Keyboard, Laptop, mouse, phone, Battery Charger/Cable Charger/adapter/laptopadapter
- Unknown policy: no detection or confidence below the frozen threshold returns `unknown`
- Integrity: record SHA-256 with `python tools/model_audit.py` before deployment

Current artifact hashes:

- `waste_detector.pt`: `96f2fe450bba2cfca6c50f9c7a519ee24b4c543b19dbe265b37013e5ef18eae5`
- `waste_detector.onnx` (320 export): `7051a9f3d46e9b50dd03cdd8fed62cbfd75a3aa412e47e7c062c6e5d91cfc7f7`
- `waste_detector_640.onnx` (accuracy profile): `9fa169131a48502ae80331e6fa1048d2c01aff7030572645704c14730f491e94`

The 6-class TFLite artifacts are not part of the demo path. `model_quantized.tflite`
is known-invalid on the target interpreter and must not be selected.

Runtime defaults for the CPU demo are `ECO_IOT_INFER_FPS=8` and
`ECO_IOT_OPENCV_THREADS=2`: camera capture/streaming stays responsive while the
worker consumes the newest frame only. Override only after measuring on target.
