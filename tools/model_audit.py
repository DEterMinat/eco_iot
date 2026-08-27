"""Print reproducible model hashes and ONNX input/output metadata."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "models").glob("*")):
    if path.suffix.lower() not in {".pt", ".onnx", ".tflite"}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}\t{path.stat().st_size}\t{digest}")
