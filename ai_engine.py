"""
ECO-Gradian IoT Edge — Lightweight AI Inference & Object Detection Engine.

Performance & Features:
- Vectorized YOLOv8 ONNX Decoder (< 2ms post-processing)
- Asynchronous Bounding Box Cache for silky-smooth 30 FPS video streaming
- Smart "Unknown" Waste handling when objects don't match known classes
- 119 E-Waste & Recyclable Class Recognition (SmartEWasteApp model)
- Low memory footprint (≤50 MB)
"""
import base64
import logging
import time
import os
import threading
import hashlib
from collections import deque
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

from config import get_config, MODELS_DIR

logger = logging.getLogger("eco_iot.ai_engine")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── 119 SmartEWasteApp Class Labels ───────────────────────────────────────────
EWASTE_119_NAMES = [
    "Air Conditioner", "Air Pump", "Amplifier", "Android Car Radio", "Antenae",
    "Appliances", "Bags", "Battery Charger", "Blender", "Bulb",
    "Cable Charger", "Calculator", "Camera", "Capacitator", "Card Reader",
    "Carpet", "Cell Battery", "Circuit Breaker", "Coffee Maker", "Cooking Pot",
    "DVD Player", "Decorder", "Deep Fryer", "Desktop DVD Writer", "Dish washer",
    "Disk Player", "Docking Station", "Extension", "External Hard Drive", "Extinguisher",
    "Fan", "Fax Machine", "Fly Swatter", "Fridge", "Gamer",
    "Gas Cylinder", "Glass", "HDPE", "Hand Mixer", "Headphone",
    "Heater", "Hot Air gun", "I Pods", "IP Phone", "Internet Cables",
    "Kettle", "Keyboard", "Lamps", "Lantern", "Laptop",
    "Laptop battery", "Laptop case", "Metals", "Microphone", "Microwave",
    "Motherboard", "Motor", "Nebulizer Compressor", "PBX", "PET",
    "PS Pad Controls", "Paperpunch", "Patch panel", "Photo Copier", "Piano",
    "Portable DIgital Drumset", "Power Cord", "Power Supply Unit", "Printer", "RAM",
    "RJ 45", "Router", "Satelite Dish", "Scanner", "Shredder",
    "Smart Watch", "Solder Gun", "Stylus Pens", "Tonner Catridge", "Video Cables",
    "adapter", "cables", "cassette deck", "computer", "dry cell batteries",
    "earphones", "iron box", "laptopadapter", "monitor", "monitor stand",
    "mouse", "phone", "plastic", "power backup", "power bank",
    "ppe", "printer accessories", "printer base", "radio", "remote",
    "rollers", "sandwich maker", "server", "server mounting kits", "socket",
    "speaker", "switch", "system unit", "tablet", "television",
    "toaster", "toner", "turn table", "typewritter", "ups",
    "vacuum cleaner", "vape", "vhf radio", "water dispenser"
]

# ── Waste Category Definitions & Color Palette (BGR for OpenCV) ───────────────
CATEGORY_CONFIG = {
    "e_waste": {
        "th_name": "ขยะอิเล็กทรอนิกส์ (E-Waste / อันตราย)",
        "color_bgr": (246, 92, 139),    # Purple #8B5CF6
        "co2_factor": 15.0,
        "bin_color": "ส้ม/แดง",
    },
    "plastic": {
        "th_name": "ขยะพลาสติก (รีไซเคิล)",
        "color_bgr": (129, 185, 16),    # Emerald #10B981
        "co2_factor": 1.5,
        "bin_color": "เหลือง",
    },
    "paper": {
        "th_name": "ขยะกระดาษ",
        "color_bgr": (11, 158, 245),    # Amber #F59E0B
        "co2_factor": 0.9,
        "bin_color": "น้ำเงิน",
    },
    "metal": {
        "th_name": "ขยะโลหะ / กระป๋อง",
        "color_bgr": (246, 130, 59),    # Blue #3B82F6
        "co2_factor": 9.0,
        "bin_color": "เหลือง",
    },
    "glass": {
        "th_name": "ขยะแก้ว / หลอดไฟ",
        "color_bgr": (212, 182, 6),     # Cyan #06B6D4
        "co2_factor": 0.3,
        "bin_color": "เขียว",
    },
    "organic": {
        "th_name": "ขยะอินทรีย์ / เศษอาหาร",
        "color_bgr": (22, 204, 132),    # Lime #84CC16
        "co2_factor": 0.5,
        "bin_color": "เขียว",
    },
    "unknown": {
        "th_name": "วัตถุไม่ระบุชนิด (Unknown)",
        "color_bgr": (148, 163, 184),   # Slate #94A3B8
        "co2_factor": 0.0,
        "bin_color": "เทา",
    },
    "general": {
        "th_name": "ขยะทั่วไป",
        "color_bgr": (184, 163, 148),   # Slate #94A3B8
        "co2_factor": 0.1,
        "bin_color": "ฟ้า",
    },
}


def map_ewaste_to_category(label_name: str) -> str:
    """Map 119 specific classes to standard waste categories."""
    lower = label_name.lower().strip()

    if "unknown" in lower:
        return "unknown"

    if any(k in lower for k in ["plastic", "hdpe", "pet", "bag", "case", "roller"]):
        return "plastic"

    if any(k in lower for k in ["metal", "cooking pot", "gas cylinder", "extinguisher", "solder"]):
        return "metal"

    if any(k in lower for k in ["glass", "bulb", "lamp", "lantern"]):
        return "glass"

    if any(k in lower for k in ["paper", "cardboard", "box", "book", "toner", "catridge"]):
        return "paper"

    if any(k in lower for k in ["food", "fruit", "organic", "plant"]):
        return "organic"

    # Default for all electronic items/devices/cables/batteries
    return "e_waste"


class AIEngine:
    """
    Lightweight Waste Classifier & Object Detection Engine.
    Features: Vectorized YOLOv8 ONNX, Async Bounding Box Cache, and Unknown object handling.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self._backend: str = "heuristic"
        self._net = None
        self._input_size = (320, 320)
        self._conf_threshold = float(getattr(self.cfg, "confidence_threshold", 0.35))
        self._nms_threshold = 0.45
        self._labels = EWASTE_119_NAMES
        self._model_sha256 = ""
        self._latencies_ms = deque(maxlen=200)

        # Asynchronous Detection Cache for 30 FPS streaming
        self._cached_detections: List[Dict[str, Any]] = []
        self._last_infer_time: float = 0.0
        self._infer_lock = threading.Lock()
        self._min_infer_interval = 0.08  # Run inference max ~12 times/sec (saves CPU for smooth streaming)

        self._load_model()

    @property
    def confidence_threshold(self) -> float:
        return self._conf_threshold

    def set_confidence(self, val: float):
        """Update confidence threshold dynamically (e.g. 0.10 to 0.95)."""
        self._conf_threshold = max(0.05, min(0.99, float(val)))
        logger.info(f"AI Engine confidence threshold updated to: {self._conf_threshold:.2f}")

    def _load_model(self):
        """Load waste_detector.onnx via OpenCV DNN or fallback."""
        if not HAS_CV2:
            self._backend = "heuristic"
            return

        model_candidates = [
            str(MODELS_DIR / "waste_detector.onnx"),
            "/root/eco_iot/models/waste_detector.onnx",
            "/home/orangepi/SmartEWasteApp/backend/models/best.onnx",
            str(MODELS_DIR / "waste_classifier.onnx"),
        ]

        for path in model_candidates:
            if os.path.isfile(path):
                try:
                    net = cv2.dnn.readNetFromONNX(path)
                    # Optimize for Orange Pi CPU
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self._net = net
                    self._backend = "yolov8_onnx"
                    try:
                        with open(path, "rb") as model_file:
                            self._model_sha256 = hashlib.sha256(model_file.read()).hexdigest()
                    except OSError:
                        self._model_sha256 = ""
                    logger.info(f"AI Engine loaded YOLOv8 ONNX: {path} (119 Classes)")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model {path}: {e}")

        self._backend = "heuristic"
        logger.info("AI Engine: using Heuristic fallback")

    # ── High-Speed Vectorized Object Detection ─────────────────────────────────

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Run object detection on frame and return list of bounding boxes + classification.
        """
        if frame is None or not HAS_CV2 or not HAS_NUMPY:
            return []

        h, w = frame.shape[:2]

        if self._backend == "yolov8_onnx" and self._net is not None:
            return self._detect_yolov8_vectorized(frame, w, h)
        return []

    def _detect_yolov8_vectorized(self, frame, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Vectorized YOLOv8 output tensor decoder (<2ms post-processing)."""
        try:
            # 1. Fast Blob creation
            # Ultralytics-compatible letterbox (preserves object geometry).
            in_w, in_h = self._input_size
            scale = min(in_w / img_w, in_h / img_h)
            resized_w, resized_h = max(1, round(img_w * scale)), max(1, round(img_h * scale))
            resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            pad_x, pad_y = (in_w - resized_w) // 2, (in_h - resized_h) // 2
            canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
            canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
            blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, self._input_size, swapRB=True, crop=False)
            self._net.setInput(blob)
            preds = self._net.forward()  # (1, 123, 2100)

            # 2. Vectorized NumPy matrix slicing
            output = preds[0].T  # (2100, 123)
            scores_matrix = output[:, 4:]  # (2100, 119)
            class_ids_arr = np.argmax(scores_matrix, axis=1)
            max_scores = np.max(scores_matrix, axis=1)

            # Vectorized threshold mask
            mask = max_scores >= self._conf_threshold

            detections = []
            if np.any(mask):
                filtered_scores = max_scores[mask]
                filtered_cids = class_ids_arr[mask]
                filtered_boxes_raw = output[mask, :4]

                cx = (filtered_boxes_raw[:, 0] - pad_x) / scale
                cy = (filtered_boxes_raw[:, 1] - pad_y) / scale
                bw = filtered_boxes_raw[:, 2]
                bh = filtered_boxes_raw[:, 3]
                bw = bw / scale
                bh = bh / scale

                lefts = np.clip(cx - 0.5 * bw, 0, img_w - 1).astype(int)
                tops = np.clip(cy - 0.5 * bh, 0, img_h - 1).astype(int)
                widths = np.clip(bw, 1, img_w).astype(int)
                heights = np.clip(bh, 1, img_h).astype(int)

                boxes = []
                confidences = []
                class_ids = []

                for l, t, wid, hei, score, cid in zip(lefts, tops, widths, heights, filtered_scores, filtered_cids):
                    if wid > 15 and hei > 15:
                        boxes.append([int(l), int(t), int(wid), int(hei)])
                        confidences.append(float(score))
                        class_ids.append(int(cid))

                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf_threshold, self._nms_threshold)
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            bx, by, bw, bh = boxes[idx]
                            cid = class_ids[idx]
                            conf = confidences[idx]
                            raw_name = self._labels[cid] if cid < len(self._labels) else f"Unknown_Object"
                            waste_cat = map_ewaste_to_category(raw_name)
                            co2 = CATEGORY_CONFIG.get(waste_cat, {}).get("co2_factor", 0.0)

                            detections.append({
                                "box": [int(bx), int(by), int(bx + bw), int(by + bh)],
                                "label": f"{raw_name}",
                                "waste_type": waste_cat,
                                "confidence": round(conf, 2),
                                "co2_offset_kg": co2,
                                "disposal_action": f"OPEN_FLAP_{waste_cat.upper()}",
                            })

            if detections:
                detections.sort(key=lambda d: d["confidence"], reverse=True)
                return detections[:5]

        except Exception as e:
            logger.error(f"YOLOv8 decode error: {e}")

        # A detector miss is unknown; never infer a class from pixels/shape.
        return []

    def _detect_unknown_saliency(self, frame, w: int, h: int) -> List[Dict[str, Any]]:
        """Detect held foreground object and mark as 'Unknown' if not matching classes."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (w * h) * 0.04
            max_area = (w * h) * 0.70

            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Filter out bottom edges (shoulders/tables)
                    if y + bh < h * 0.95 or area > (w * h) * 0.08:
                        candidates.append((x, y, bw, bh, area))

            if candidates:
                candidates.sort(key=lambda c: c[4], reverse=True)
                x, y, bw, bh, _ = candidates[0]
                return [{
                    "box": [int(x), int(y), int(x + bw), int(y + bh)],
                    "label": "Unknown",
                    "waste_type": "unknown",
                    "confidence": 0.45,
                    "co2_offset_kg": 0.0,
                    "disposal_action": "CHECK_MANUAL",
                }]
        except Exception:
            pass

        return []

    # ── Smooth Stream Annotation (Asynchronous Bounding Box Cache) ─────────────

    def annotate_frame_smooth(self, frame):
        """
        Annotate video stream frame smoothly with cached detections to prevent stutter.
        """
        if frame is None or not HAS_CV2:
            return frame

        now = time.time()
        # Throttled async inference trigger
        if now - self._last_infer_time >= self._min_infer_interval:
            if self._infer_lock.acquire(blocking=False):
                try:
                    # Run inference on frame copy
                    self._cached_detections = self.detect_objects(frame)
                    self._last_infer_time = now
                finally:
                    self._infer_lock.release()

        # Render cached bounding boxes
        return self.annotate_frame(frame, detections=self._cached_detections)

    def annotate_frame(self, frame, detections: Optional[List[Dict]] = None):
        """Draw modern bounding boxes and category pills on frame."""
        if frame is None or not HAS_CV2:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if detections is None:
            detections = self.detect_objects(annotated)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            waste_type = det.get("waste_type", "unknown")
            conf = det.get("confidence", 0.0)
            label = det.get("label", "Unknown")
            co2 = det.get("co2_offset_kg", 0.0)

            cfg = CATEGORY_CONFIG.get(waste_type, CATEGORY_CONFIG["unknown"])
            bgr = cfg["color_bgr"]

            # 1. Semi-transparent bounding box background
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
            cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)

            # 2. Main rectangle border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            # 3. Corner Brackets
            corner_len = min(int((x2 - x1) * 0.18), int((y2 - y1) * 0.18), 24)
            thk = 4
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), bgr, thk)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), bgr, thk)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), bgr, thk)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), bgr, thk)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), bgr, thk)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), bgr, thk)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), bgr, thk)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), bgr, thk)

            # 4. Header Pill Badge
            tag_text = f"[{waste_type.upper()}] {label} ({int(conf * 100)}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, 1)

            pill_y1 = max(0, y1 - th - 12)
            pill_y2 = y1
            pill_x2 = min(w, x1 + tw + 16)

            cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), bgr, -1)
            cv2.putText(
                annotated,
                tag_text,
                (x1 + 8, pill_y2 - 6),
                font,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # 5. Bottom Carbon Badge (Only if known waste)
            if co2 > 0:
                co2_text = f"CO2: -{co2:.1f}kg"
                (cw, ch), _ = cv2.getTextSize(co2_text, font, 0.42, 1)
                bot_y1 = y2
                bot_y2 = min(h, y2 + ch + 10)
                bot_x2 = min(w, x1 + cw + 14)

                cv2.rectangle(annotated, (x1, bot_y1), (bot_x2, bot_y2), (20, 20, 20), -1)
                cv2.rectangle(annotated, (x1, bot_y1), (bot_x2, bot_y2), bgr, 1)
                cv2.putText(
                    annotated,
                    co2_text,
                    (x1 + 6, bot_y2 - 4),
                    font,
                    0.42,
                    (200, 255, 200),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Top Status HUD Bar
        cv2.rectangle(annotated, (0, 0), (w, 28), (15, 23, 42), -1)
        hud_text = f"ECO-Gradian AI | YOLOv8 (119 Classes) | Conf: {int(self._conf_threshold*100)}% | Items: {len(detections)}"
        cv2.putText(annotated, hud_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (16, 185, 129), 1, cv2.LINE_AA)

        return annotated

    # ── Overall Image Prediction ──────────────────────────────────────────────

    def predict(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """Classify waste and detect objects in JPEG image."""
        t0 = time.time()
        img = self._decode_jpeg(jpeg_bytes)

        detections = []
        if img is not None:
            detections = self.detect_objects(img)

        if detections:
            primary = detections[0]
            raw_label = primary.get("label", "unknown")
            label = primary["waste_type"]
            conf = max(0.0, min(1.0, float(primary["confidence"])))
            co2 = primary["co2_offset_kg"]
            disposal = primary["disposal_action"] if label != "unknown" else None
            class_id = self._labels.index(raw_label) if raw_label in self._labels else None
        else:
            label, conf = "unknown", 0.0
            co2 = 0.0
            disposal = None
            raw_label = "unknown"
            class_id = None

        latency = (time.time() - t0) * 1000

        self._latencies_ms.append(latency)
        return {
            "status": "detected" if label != "unknown" else "unknown",
            "class": label,
            "item_class": raw_label,
            "item_label": raw_label,
            "class_id": class_id,
            "box": detections[0].get("box") if detections else None,
            "detections": detections,
            "waste_type": label,
            "confidence": round(conf, 4),
            "image_hash": hashlib.sha256(jpeg_bytes).hexdigest()[:16],
            "latency_ms": round(latency, 1),
            "backend": self._backend,
            "co2_offset_kg": co2,
            "disposal_action": disposal,
            "model": {"backend": self._backend, "version": "demo-2026-08-28", "sha256": self._model_sha256},
            "detected_objects": detections,
            "category_info": CATEGORY_CONFIG.get(label, CATEGORY_CONFIG["unknown"]),
        }

    @staticmethod
    def _decode_jpeg(jpeg_bytes: bytes):
        if not HAS_CV2 or not HAS_NUMPY:
            return None
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def latency_stats(self) -> Dict[str, float]:
        values = sorted(self._latencies_ms)
        if not values:
            return {"p50_ms": 0.0, "p95_ms": 0.0}
        return {"p50_ms": round(values[len(values)//2], 1), "p95_ms": round(values[min(len(values)-1, int(len(values)*.95))], 1)}
