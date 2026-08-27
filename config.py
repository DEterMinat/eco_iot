"""
ECO-Gradian IoT Edge Device — Configuration & RAM Constraints.

All limits enforced at runtime to guarantee ≤1GB total RAM usage.
"""
import os
import secrets
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
KEYS_FILE = DATA_DIR / "api_keys.json"

# Ensure directories exist on import
for d in (MODELS_DIR, LOGS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EdgeConfig:
    """Immutable runtime configuration for the IoT edge device."""

    # ── Network ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # Single worker to minimise RAM

    # ── Camera (USB / CSI / RTSP) ──────────────────────────────────────────────
    camera_index: int = 0          # /dev/video0 by default
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    jpeg_quality: int = 80         # Lower = smaller frames = less RAM

    # ── AI Model ───────────────────────────────────────────────────────────────
    model_path: str = str(MODELS_DIR / "waste_detector.onnx")
    confidence_threshold: float = 0.30
    enable_heuristic_fallback: bool = False   # False = only run 119-class YOLO model
    enable_contour_fallback: bool = False     # False = disable heuristic plastic/glass/paper guessing
    input_size: tuple = (320, 320)            # YOLOv8 input size
    class_labels: tuple = (
        "e_waste", "plastic", "paper", "metal", "glass", "organic", "general"
    )

    # ── RAM Guard ──────────────────────────────────────────────────────────────
    max_ram_mb: int = 900           # Hard ceiling — process restarts above this
    warn_ram_mb: int = 700          # Log warning above this threshold
    gc_interval_sec: int = 60       # Garbage-collection sweep period
    frame_buffer_max: int = 3       # Max JPEG frames kept in memory ring buffer

    # ── Log & Cache Cleanup ────────────────────────────────────────────────────
    log_max_size_mb: int = 20       # Rotate & prune when total exceeds this
    log_max_age_hours: int = 48     # Delete log files older than 2 days
    cache_max_size_mb: int = 50     # Prune inference cache above this
    cleanup_interval_sec: int = 300 # Run cleanup every 5 minutes

    # ── API Key Authentication ─────────────────────────────────────────────────
    require_api_key: bool = True    # Reject unauthenticated requests
    master_key: str = field(default_factory=lambda: os.environ.get(
        "ECO_IOT_MASTER_KEY", "ek_master_" + secrets.token_hex(12)
    ))

    # ── Upstream ECO-Gradian Backend ───────────────────────────────────────────
    upstream_url: Optional[str] = os.environ.get(
        "ECO_UPSTREAM_URL", "http://localhost:8000"
    )
    upstream_sync_interval: int = 30  # Sync telemetry every N seconds

    # ── Carbon Offset Factors (kg CO₂e per item) ──────────────────────────────
    carbon_factors: dict = field(default_factory=lambda: {
        "plastic": 1.5, "paper": 0.9, "metal": 9.0,
        "glass": 0.3, "organic": 0.5, "e_waste": 15.0, "general": 0.1,
    })


# Singleton config — override via env vars or CLI
_config: Optional[EdgeConfig] = None


def get_config(**overrides) -> EdgeConfig:
    """Return (or create) the global EdgeConfig singleton."""
    global _config
    if _config is None or overrides:
        _config = EdgeConfig(**overrides)
    return _config
