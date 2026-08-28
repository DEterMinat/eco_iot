"""
ECO-Gradian IoT Edge — USB Camera Manager.

Low-RAM OpenCV pipeline optimised for Orange Pi / Raspberry Pi:
- Single VideoCapture with V4L2 backend preference
- Ring buffer of N frames (default 3) to bound memory
- Auto-reconnect on camera disconnect
- Resolution capped at 640×480 to save bandwidth and RAM
"""
import base64
import logging
import threading
import time
from collections import deque
from typing import Optional, Union

logger = logging.getLogger("eco_iot.camera")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not installed — camera will run in simulation mode")


class CameraManager:
    """Thread-safe USB / CSI camera wrapper with bounded frame buffer."""

    def __init__(
        self,
        index: int = 0,
        source_url: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        jpeg_quality: int = 80,
        buffer_size: int = 3,
    ):
        self.index = index
        self.source_url = source_url.strip() if source_url else None
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._cap: Optional["cv2.VideoCapture"] = None
        self._lock = threading.Lock()
        self._buffer: deque = deque(maxlen=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_error: Optional[str] = None
        self._latest_bgr = None
        self._last_frame_at = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the camera device. Returns True on success."""
        if not HAS_CV2:
            logger.info("📷 Camera (simulation mode — no OpenCV)")
            return True

        with self._lock:
            if self._cap and self._cap.isOpened():
                return True

            source: Union[int, str] = self.source_url or self.index
            # V4L2 is for local devices; network URLs need the generic backend.
            backends = ([cv2.CAP_ANY] if self.source_url else
                        ([cv2.CAP_V4L2, cv2.CAP_ANY] if hasattr(cv2, "CAP_V4L2") else [cv2.CAP_ANY]))

            for backend in backends:
                try:
                    cap = cv2.VideoCapture(source, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        # Reduce internal buffer to save RAM
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self._cap = cap
                        self._last_error = None
                        logger.info(
                            f"✅ Camera opened: /dev/video{self.index} "
                            f"({self.width}×{self.height} @{self.fps}fps, backend={backend})"
                        )
                        return True
                    cap.release()
                except Exception as e:
                    logger.debug(f"Backend {backend} failed: {e}")

            self._last_error = f"Cannot open camera source {self.source_url or self.index}"
            logger.error(f"❌ {self._last_error}")
            return False

    def close(self):
        """Release camera resources."""
        self.stop_capture()
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
                logger.info("📷 Camera released")

    def start_capture(self):
        """Start background thread that continuously grabs frames."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CamCapture")
        self._thread.start()

    def stop_capture(self):
        """Stop background capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    # ── Frame Access ───────────────────────────────────────────────────────────

    def grab_frame_jpeg(self) -> Optional[bytes]:
        """Return the latest JPEG-encoded frame bytes, or None."""
        if self._buffer:
            return self._buffer[-1]

        # Fallback: direct synchronous grab
        return self._read_one_frame()

    def grab_frame_bgr(self):
        """Return raw OpenCV BGR numpy array frame from cache without blocking."""
        with self._lock:
            if self._latest_bgr is not None:
                return self._latest_bgr.copy()
        if not HAS_CV2 or not self._cap:
            return None
        with self._lock:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self._latest_bgr = frame
                self._last_frame_at = time.monotonic()
                return frame.copy()
        return None

    def grab_frame_base64(self) -> Optional[str]:
        """Return the latest frame as a base64-encoded JPEG string."""
        jpeg = self.grab_frame_jpeg()
        if jpeg:
            return base64.b64encode(jpeg).decode("ascii")
        return None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        if not HAS_CV2:
            return True  # simulation
        return self._cap is not None and self._cap.isOpened()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def last_frame_age_ms(self) -> Optional[float]:
        """Age of the newest successfully decoded frame."""
        if self._last_frame_at <= 0:
            return None
        return round((time.monotonic() - self._last_frame_at) * 1000.0, 1)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _read_one_frame(self) -> Optional[bytes]:
        """Single synchronous frame read (for on-demand capture)."""
        if not HAS_CV2 or not self._cap:
            return None

        with self._lock:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._last_error = "Frame grab failed"
                # Mark a dead network/device stream closed so the capture loop
                # can reopen it with backoff instead of spinning on a stale
                # VideoCapture handle.
                try:
                    self._cap.release()
                finally:
                    self._cap = None
                return None

            # Resize if needed (should already be set via CAP_PROP)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self._latest_bgr = frame
            self._last_frame_at = time.monotonic()
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                self._frame_count += 1
                return buf.tobytes()

        return None

    def _capture_loop(self):
        """Background loop: grab frames into ring buffer."""
        reconnect_delay = 1.0
        interval = 1.0 / max(self.fps, 1)

        while self._running:
            jpeg = self._read_one_frame()
            if jpeg:
                self._buffer.append(jpeg)
                reconnect_delay = 1.0
                time.sleep(interval)
            else:
                # Camera lost — attempt reconnect with backoff
                logger.warning(f"📷 Camera lost, reconnecting in {reconnect_delay:.0f}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
                self.open()

    @staticmethod
    def _synthetic_jpeg() -> bytes:
        """1×1 green pixel JPEG for simulation / offline fallback."""
        # Minimal valid JPEG (green pixel)
        return bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7B, 0x40,
            0x1F, 0xFF, 0xD9,
        ])
