"""Export 320/384/640 ONNX profiles from the frozen YOLOv8 weights.

Run on a build machine with ultralytics installed; deployment still uses the
profile selected by measured golden-set accuracy and latency.
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
model = YOLO(str(root / "models" / "waste_detector.pt"))
for size in (320, 384, 640):
    model.export(format="onnx", imgsz=size, opset=12, simplify=False, nms=False,
                 project=str(root / "models"), name=f"waste_detector_{size}", exist_ok=True)
    generated = root / "models" / "waste_detector.onnx"
    shutil.copy2(generated, root / "models" / f"waste_detector_{size}.onnx")
    if size == 320:
        shutil.copy2(generated, root / "models" / "waste_detector.onnx")
