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
- `waste_detector.onnx`: `eaed98673cdd9f42c9d54225e7a0d85c9107d14908fbac7e0328b9ee17db918c`

The 6-class TFLite artifacts are not part of the demo path. `model_quantized.tflite`
is known-invalid on the target interpreter and must not be selected.
