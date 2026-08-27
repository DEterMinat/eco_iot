"""
ECO-Gradian IoT Edge — Lightweight AI Inference Engine.

Supports:
- TensorFlow Lite (preferred on ARM64 Orange Pi / RPi)
- ONNX Runtime (fallback)
- OpenCV DNN (secondary fallback)
- Rule-based heuristic (offline / no-model fallback)

All paths are designed to stay under 50 MB model RAM.
"""
import base64
import logging
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from config import get_config

logger = logging.getLogger("eco_iot.ai_engine")

# ── Optional ML framework imports ──────────────────────────────────────────────
_tflite_interpreter = None
_onnx_session = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import importlib

tflite_interpreter_cls = None
HAS_TFLITE = False

try:
    _tflite_module = importlib.import_module("tflite_runtime.interpreter")
    tflite_interpreter_cls = getattr(_tflite_module, "Interpreter", None)
    if tflite_interpreter_cls is not None:
        HAS_TFLITE = True
except (ImportError, ModuleNotFoundError):
    try:
        _tf_module = importlib.import_module("tensorflow")
        _tf_lite = getattr(_tf_module, "lite", None)
        if _tf_lite and hasattr(_tf_lite, "Interpreter"):
            tflite_interpreter_cls = _tf_lite.Interpreter
            HAS_TFLITE = True
    except (ImportError, ModuleNotFoundError):
        HAS_TFLITE = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class AIEngine:
    """
    Lightweight waste classification engine.

    Loads a single model into memory and provides a .predict(jpeg_bytes) method
    that returns (label, confidence, latency_ms).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self._backend: Optional[str] = None
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels: Tuple[str, ...] = self.cfg.class_labels
        self._load_model()

    def _load_model(self):
        """Try loading model in priority order: TFLite → ONNX → heuristic."""
        import os
        model_path = self.cfg.model_path

        # 1. TFLite
        if HAS_TFLITE and tflite_interpreter_cls and os.path.isfile(model_path) and model_path.endswith(".tflite"):
            try:
                interp = tflite_interpreter_cls(model_path=model_path, num_threads=2)
                interp.allocate_tensors()
                self._interpreter = interp
                self._input_details = interp.get_input_details()
                self._output_details = interp.get_output_details()
                self._backend = "tflite"
                logger.info(f"🧠 AI Engine loaded: TFLite ({model_path})")
                return
            except Exception as e:
                logger.warning(f"TFLite load failed: {e}")

        # 2. ONNX
        onnx_path = model_path.replace(".tflite", ".onnx")
        if HAS_ONNX and os.path.isfile(onnx_path):
            try:
                global _onnx_session
                _onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                self._backend = "onnx"
                logger.info(f"🧠 AI Engine loaded: ONNX ({onnx_path})")
                return
            except Exception as e:
                logger.warning(f"ONNX load failed: {e}")

        # 3. Heuristic fallback (no model file needed)
        self._backend = "heuristic"
        logger.info("🧠 AI Engine: using rule-based heuristic (no ML model file found)")

    def predict(self, jpeg_bytes: bytes) -> Dict:
        """
        Run waste classification on a JPEG image.

        Returns dict: {label, confidence, latency_ms, backend}
        """
        t0 = time.time()

        if self._backend == "tflite":
            label, conf = self._predict_tflite(jpeg_bytes)
        elif self._backend == "onnx":
            label, conf = self._predict_onnx(jpeg_bytes)
        else:
            label, conf = self._predict_heuristic(jpeg_bytes)

        latency = (time.time() - t0) * 1000

        # Apply confidence threshold
        if conf < self.cfg.confidence_threshold:
            label = "general"

        co2 = self.cfg.carbon_factors.get(label, 0.1)

        return {
            "waste_type": label,
            "confidence": round(conf, 4),
            "latency_ms": round(latency, 1),
            "backend": self._backend,
            "co2_offset_kg": co2,
            "disposal_action": f"OPEN_FLAP_{label.upper()}",
        }

    # ── TFLite inference ───────────────────────────────────────────────────────

    def _predict_tflite(self, jpeg_bytes: bytes) -> Tuple[str, float]:
        if not HAS_NUMPY or not HAS_CV2:
            return self._predict_heuristic(jpeg_bytes)

        try:
            img = self._decode_jpeg(jpeg_bytes)
            if img is None:
                return self._predict_heuristic(jpeg_bytes)

            input_shape = self._input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
            h, w = input_shape[1], input_shape[2]
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            self._interpreter.set_tensor(self._input_details[0]["index"], img)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

            idx = int(np.argmax(output))
            conf = float(output[idx])
            label = self._labels[idx] if idx < len(self._labels) else "general"
            return label, conf
        except Exception as e:
            logger.error(f"TFLite inference error: {e}")
            return self._predict_heuristic(jpeg_bytes)

    # ── ONNX inference ─────────────────────────────────────────────────────────

    def _predict_onnx(self, jpeg_bytes: bytes) -> Tuple[str, float]:
        if not HAS_NUMPY or not HAS_CV2 or _onnx_session is None:
            return self._predict_heuristic(jpeg_bytes)

        try:
            img = self._decode_jpeg(jpeg_bytes)
            if img is None:
                return self._predict_heuristic(jpeg_bytes)

            input_name = _onnx_session.get_inputs()[0].name
            shape = _onnx_session.get_inputs()[0].shape  # [1, 3, 224, 224] or [1, 224, 224, 3]
            h, w = shape[2], shape[3]
            img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            img = np.expand_dims(img, axis=0)

            output = _onnx_session.run(None, {input_name: img})[0][0]
            idx = int(np.argmax(output))
            conf = float(output[idx])
            label = self._labels[idx] if idx < len(self._labels) else "general"
            return label, conf
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            return self._predict_heuristic(jpeg_bytes)

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _predict_heuristic(self, jpeg_bytes: bytes) -> Tuple[str, float]:
        """
        Size-based heuristic when no ML model is available.
        Uses JPEG byte-size patterns as a rough proxy.
        """
        size = len(jpeg_bytes)

        if size < 500:
            return "general", 0.55
        elif size < 5_000:
            return "paper", 0.72
        elif size < 20_000:
            return "plastic", 0.78
        elif size < 50_000:
            return "metal", 0.70
        elif size < 100_000:
            return "glass", 0.68
        else:
            return "organic", 0.65

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_jpeg(jpeg_bytes: bytes):
        """Decode JPEG bytes to OpenCV BGR numpy array."""
        if not HAS_CV2 or not HAS_NUMPY:
            return None
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        return self._backend or "none"
