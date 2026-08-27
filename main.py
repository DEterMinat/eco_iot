#!/usr/bin/env python3
"""
=============================================================================
ECO-Gradian IoT Edge Server — main.py
=============================================================================
Production-ready FastAPI edge server for headless Linux IoT devices.
Designed for <=1 GB RAM (Orange Pi, RPi, Docker containers).

Multi-layer RAM enforcement:
  1. Application: MemoryGuard daemon (GC + cache purge + auto-restart)
  2. systemd:     MemoryMax=1024M
  3. Docker:      --memory=1g --memory-swap=1g
  4. Linux:       RLIMIT_AS via resource module

All routes are crash-isolated — a single endpoint failure never kills the server.

Usage:
  python main.py                            # Start with defaults
  python main.py --camera 0 --port 8080     # Specify camera & port
  python main.py --no-camera                # Headless, no camera
  python main.py --generate-key "Lab #3"    # Generate a new API key
  python main.py --list-keys                # Show all registered keys
=============================================================================
"""
import argparse
import base64
import gc
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config, LOGS_DIR, DATA_DIR, BASE_DIR
from key_manager import generate_key, validate_key, list_keys
from memory_guard import MemoryGuardDaemon, get_process_rss_mb, force_gc, get_system_mem_mb
from camera import CameraManager
from ai_engine import AIEngine

# ── Logging setup (auto-rotating file + stdout, no GUI) ───────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _log_handlers.append(
        logging.handlers.RotatingFileHandler(
            LOGS_DIR / "eco_iot.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
    )
except OSError:
    pass  # Read-only filesystem (Docker tmpfs edge case)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("eco_iot")

import asyncio

# ── FastAPI imports ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException, Response
    from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
    import uvicorn
except ImportError:
    logger.error("FastAPI/Uvicorn not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# ── Graceful shutdown handler ─────────────────────────────────────────────────
_shutdown_event = False

def _signal_handler(sig, frame):
    global _shutdown_event
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    _shutdown_event = True
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# CLI Commands
# =============================================================================

def handle_cli():
    parser = argparse.ArgumentParser(description="ECO-Gradian IoT Edge Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ECO_IOT_PORT", "8080")))
    parser.add_argument("--camera", type=int, default=int(os.environ.get("ECO_IOT_CAMERA", "0")))
    parser.add_argument("--generate-key", type=str, metavar="LABEL")
    parser.add_argument("--list-keys", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-auth", action="store_true")
    args = parser.parse_args()

    if args.generate_key:
        raw_key, meta = generate_key(label=args.generate_key)
        print("=" * 60)
        print("  ECO-Gradian IoT -- New API Key Generated")
        print("=" * 60)
        print(f"  Label      : {meta['label']}")
        print(f"  API Key    : {raw_key}")
        print(f"  Scopes     : {', '.join(meta['scopes'])}")
        print(f"  Created    : {meta['created_at']}")
        print("=" * 60)
        print("  WARNING: Save this key -- it will NOT be shown again!")
        print("=" * 60)
        sys.exit(0)

    if args.list_keys:
        keys = list_keys()
        print(f"\nRegistered API Keys ({len(keys)} total):\n")
        for k in keys:
            status = "ACTIVE" if k["active"] else "REVOKED"
            print(
                f"  [{k['key_id']}] {k['label']:30s} {status}  "
                f"| Used: {k['request_count']}x | Last: {k['last_used'] or 'never'}"
            )
        print()
        sys.exit(0)

    return args


# =============================================================================
# Application Factory
# =============================================================================

def create_app(args) -> FastAPI:
    cfg = get_config(
        port=args.port,
        camera_index=args.camera,
        require_api_key=not args.no_auth,
    )

    app = FastAPI(
        title="ECO-Gradian IoT Edge Server",
        version="2.0.0",
        docs_url=None,     # No Swagger UI on headless IoT (saves RAM)
        redoc_url=None,     # No ReDoc on headless IoT
        openapi_url=None,   # No OpenAPI schema served
    )

    # ── Singletons (lazy-init, crash-resilient) ────────────────────────────────
    camera: Optional[CameraManager] = None
    ai: Optional[AIEngine] = None
    mem_guard: Optional[MemoryGuardDaemon] = None
    started_at = time.monotonic()

    # ── Startup ────────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        nonlocal camera, ai, mem_guard

        logger.info("=" * 55)
        logger.info("  ECO-Gradian IoT Edge Server Starting")
        logger.info(f"  Port:     {cfg.port}")
        logger.info(f"  Camera:   {'Disabled' if args.no_camera else f'/dev/video{cfg.camera_index}'}")
        logger.info(f"  Auth:     {'Enabled' if cfg.require_api_key else 'Disabled'}")
        logger.info(f"  RAM Ceil: {cfg.max_ram_mb} MB")
        logger.info(f"  PID:      {os.getpid()}")

        # Detect Docker / container environment
        in_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        logger.info(f"  Docker:   {'Yes' if in_docker else 'No'}")
        logger.info("=" * 55)

        # AI Engine (crash-isolated)
        try:
            ai = AIEngine(cfg)
            logger.info(f"  AI:       {ai.backend_name}")
        except Exception as e:
            logger.error(f"AI Engine init failed (non-fatal): {e}")

        # Camera (crash-isolated)
        if not args.no_camera:
            try:
                camera = CameraManager(
                    index=cfg.camera_index,
                    width=cfg.camera_width,
                    height=cfg.camera_height,
                    fps=cfg.camera_fps,
                    jpeg_quality=cfg.jpeg_quality,
                    buffer_size=cfg.frame_buffer_max,
                )
                camera.open()
                camera.start_capture()
            except Exception as e:
                logger.error(f"Camera init failed (non-fatal): {e}")
                camera = None

        # Memory Guard
        try:
            mem_guard = MemoryGuardDaemon(cfg)
            mem_guard.start()
        except Exception as e:
            logger.error(f"MemoryGuard init failed (non-fatal): {e}")

        # Force initial GC after startup
        force_gc()

        # Auto-generate a default API key if empty
        try:
            keys = list_keys()
            if not keys:
                raw_key, _ = generate_key(label="auto-default")
                logger.info(f"Auto-generated API key: {raw_key}")
        except Exception:
            pass

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down...")
        if camera:
            try:
                camera.close()
            except Exception:
                pass
        if ai:
            try:
                ai.close()
            except Exception:
                pass
        if mem_guard:
            try:
                mem_guard.stop()
            except Exception:
                pass
        logger.info("Shutdown complete.")

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )

    # ── Auth Middleware ─────────────────────────────────────────────────────────
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        # Skip auth for health/root
        if path in ("/", "/health"):
            return await call_next(request)

        if cfg.require_api_key:
            api_key = (
                request.headers.get("x-api-key")
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                or request.query_params.get("api_key")
            )
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing API key"},
                )
            meta = validate_key(api_key)
            if meta is None:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid or revoked API key"},
                )
            request.state.key_label = meta.get("label", "?")

        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Request error {path}: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "internal_error"},
            )

    # ── Routes (all crash-isolated) ────────────────────────────────────────────

    @app.get("/")
    async def root():
        return {
            "service": "ECO-Gradian IoT Edge",
            "version": "2.0.0",
            "ai": ai.backend_name if ai else "offline",
            "camera": (camera.is_open if camera else False),
            "ram_mb": round(get_process_rss_mb(), 1),
            "pid": os.getpid(),
        }

    @app.get("/health")
    async def health():
        try:
            rss = get_process_rss_mb()
            sys_mem = get_system_mem_mb()
            return {
                "status": "healthy" if rss < cfg.max_ram_mb else "degraded",
                "process_ram_mb": round(rss, 1),
                "ram_limit_mb": cfg.max_ram_mb,
                "system_ram_total_mb": round(sys_mem.get("total_mb", 0), 1),
                "system_ram_available_mb": round(sys_mem.get("available_mb", 0), 1),
                "system_ram_used_pct": sys_mem.get("used_pct", 0),
                "ai_backend": ai.backend_name if ai else "offline",
                "model_hash": ai.model_sha256 if ai else "",
                "model_version": "demo-2026-08-28",
                "camera_open": camera.is_open if camera else False,
                "frames_captured": camera.frame_count if camera else 0,
                "actual_fps": round((camera.frame_count / max(time.monotonic() - started_at, 1e-3)) if camera else 0.0, 2),
                "inference_p50_ms": ai.latency_stats["p50_ms"] if ai else 0.0,
                "inference_p95_ms": ai.latency_stats["p95_ms"] if ai else 0.0,
                "inference_fps_limit": ai.inference_fps_limit if ai else 0.0,
                "uptime_sec": round(time.monotonic() - started_at, 1),
                "pid": os.getpid(),
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    @app.post("/predict")
    async def predict(request: Request):
        if not ai:
            raise HTTPException(status_code=503, detail="AI engine not available")

        try:
            content_type = request.headers.get("content-type", "")

            if "image/jpeg" in content_type or "application/octet-stream" in content_type:
                jpeg_bytes = await request.body()
            else:
                body = await request.json()
                b64 = body.get("image_base64", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                jpeg_bytes = base64.b64decode(b64)

            if not jpeg_bytes:
                raise HTTPException(status_code=400, detail="Empty image")

            result = ai.predict(jpeg_bytes)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["device_ram_mb"] = round(get_process_rss_mb(), 1)
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"/predict error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    @app.get("/capture")
    async def capture():
        if not camera or not camera.is_open:
            raise HTTPException(status_code=503, detail="Camera not available")
        if not ai:
            raise HTTPException(status_code=503, detail="AI engine not available")

        try:
            jpeg = camera.grab_frame_jpeg()
            if not jpeg:
                raise HTTPException(status_code=503, detail="Frame grab failed")

            result = ai.predict(jpeg)
            result["frame_bytes"] = len(jpeg)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"/capture error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/frame")
    async def live_frame():
        """Single static JPEG snapshot."""
        if not camera or not camera.is_open:
            raise HTTPException(status_code=503, detail="Camera not available")

        try:
            jpeg = camera.grab_frame_jpeg()
            if not jpeg:
                raise HTTPException(status_code=503, detail="No frame")
            return Response(content=jpeg, media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stream")
    @app.get("/video_feed")
    async def video_stream(request: Request):
        """Continuous real-time MJPEG video stream with optional AI bounding box overlay."""
        if not camera or not camera.is_open:
            raise HTTPException(status_code=503, detail="Camera not available")

        # Query param ?annotate=1 (default true)
        annotate_flag = request.query_params.get("annotate", "1").lower() in ("1", "true", "yes")

        async def _mjpeg_generator():
            try:
                import cv2
                while True:
                    if await request.is_disconnected():
                        break

                    if annotate_flag and ai:
                        # Grab cached BGR frame and draw smooth bounding boxes
                        bgr = camera.grab_frame_bgr()
                        if bgr is not None:
                            annotated = ai.annotate_frame_smooth(bgr)
                            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            jpeg = buf.tobytes() if ok else None
                        else:
                            jpeg = camera.grab_frame_jpeg()
                    else:
                        jpeg = camera.grab_frame_jpeg()

                    if jpeg:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                            + jpeg + b"\r\n"
                        )
                    await asyncio.sleep(0.04)  # ~25 FPS
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Stream error: {e}")

        return StreamingResponse(
            _mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/view", response_class=HTMLResponse)
    @app.get("/live", response_class=HTMLResponse)
    async def live_view_page(request: Request):
        """Interactive real-time live camera & AI object detection dashboard."""
        api_key = request.query_params.get("api_key", "")
        key_param = f"&api_key={api_key}" if api_key else ""
        first_key_param = f"?api_key={api_key}" if api_key else ""

        html = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECO-Gradian IoT - AI Object Detection & Live Stream</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #090D16;
            color: #F1F5F9;
            font-family: 'Outfit', 'Noto Sans Thai', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 1040px;
            margin-bottom: 14px;
        }}
        .logo {{
            font-size: 20px;
            font-weight: 700;
            color: #10B981;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.5px;
        }}
        .header-actions {{ display: flex; align-items: center; gap: 8px; }}
        .badge {{
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .badge .dot {{
            width: 8px; height: 8px; background: #10B981; border-radius: 50%;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.9); opacity: 0.7; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
            100% {{ transform: scale(0.9); opacity: 0.7; }}
        }}
        .stream-card {{
            position: relative;
            background: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 18px;
            overflow: hidden;
            width: 100%;
            max-width: 1040px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }}
        .video-box {{
            width: 100%;
            aspect-ratio: 4 / 3;
            background: #020617;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .video-box img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        .video-box::after {{ content: 'LIVE • AI OVERLAY'; position: absolute; top: 14px; right: 14px; padding: 5px 9px; border-radius: 999px; background: rgba(2,6,23,.75); color: #6EE7B7; border: 1px solid rgba(110,231,183,.35); font-size: 10px; font-weight: 700; letter-spacing: .7px; pointer-events: none; }}
        .stream-bar {{
            padding: 10px 16px;
            background: #0B1220;
            border-top: 1px solid #1E293B;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }}
        .tag-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .tag {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
            color: #94A3B8;
        }}
        .tag.active {{ color: #38BDF8; border-color: rgba(56, 189, 248, 0.3); }}
        .btn-toggle {{
            background: #1E293B;
            color: #F8FAFC;
            border: 1px solid #334155;
            padding: 6px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .btn-toggle:hover {{ background: #334155; }}
        .btn-toggle.on {{ background: #059669; color: #fff; border-color: #10B981; }}
        
        .dashboard-grid {{
            width: 100%;
            max-width: 1040px;
            margin-top: 14px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }}
        .card {{
            background: #0F172A;
            border: 1px solid #1E293B;
            padding: 16px 18px;
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-label {{ font-size: 12px; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
        .card-val {{ font-size: 19px; font-weight: 700; color: #F8FAFC; }}
        .metric-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
        .metric {{ background: #111C2E; border: 1px solid #23324A; border-radius: 10px; padding: 9px; }}
        .metric b {{ display: block; color: #F8FAFC; font-size: 15px; margin-top: 3px; }}
        .confidence-meter {{ height: 7px; background: #1E293B; border-radius: 99px; overflow: hidden; margin-top: 9px; }}
        .confidence-meter span {{ display: block; width: 35%; height: 100%; background: linear-gradient(90deg,#F59E0B,#10B981); border-radius: inherit; }}
        .result-main {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
        .result-title {{ font-size: 22px; font-weight: 700; color: #F8FAFC; }}
        .result-subtitle {{ color: #94A3B8; font-size: 12px; margin-top: 3px; }}
        .result-confidence {{ font-size: 24px; font-weight: 700; color: #6EE7B7; }}
        @media (max-width: 720px) {{ body {{ padding: 10px; }} .header {{ align-items: flex-start; gap: 10px; }} .logo {{ font-size: 16px; }} .header-actions {{ flex-wrap: wrap; justify-content: flex-end; }} .stream-bar {{ padding: 8px 10px; }} .dashboard-grid {{ grid-template-columns: 1fr; }} .metric-row {{ grid-template-columns: 1fr 1fr; }} }}
        .btn-action {{
            background: #10B981;
            color: #022C22;
            font-weight: 700;
            border: none;
            padding: 13px 20px;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        .btn-action:hover {{ background: #059669; color: #fff; transform: translateY(-1px); }}
        .btn-action:active {{ transform: translateY(0); }}
        
        .result-box {{
            margin-top: 10px;
            padding: 10px 14px;
            border-radius: 10px;
            background: #020617;
            border: 1px solid #1E293B;
            font-size: 13px;
            color: #E2E8F0;
            line-height: 1.5;
        }}
        .category-pill {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            margin-right: 4px;
        }}
        .pill-plastic {{ background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }}
        .pill-paper {{ background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }}
        .pill-metal {{ background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }}
        .pill-glass {{ background: rgba(6, 182, 212, 0.2); color: #22D3EE; border: 1px solid #06B6D4; }}
        .pill-organic {{ background: rgba(132, 204, 22, 0.2); color: #A3E635; border: 1px solid #84CC16; }}
        .pill-e_waste {{ background: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid #8B5CF6; }}
        .pill-general {{ background: rgba(148, 163, 184, 0.2); color: #CBD5E1; border: 1px solid #94A3B8; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🌿 ECO-Gradian IoT Edge</div>
        <div class="header-actions">
            <button id="toggle-box-btn" class="btn-toggle on" onclick="toggleAnnotation()">🎯 กล่อง AI: เปิด</button>
            <div class="badge"><span class="dot"></span> LIVE 25 FPS</div>
        </div>
    </div>

    <div class="stream-card">
        <div class="video-box">
            <img id="stream-img" src="/stream?annotate=1{key_param}" alt="Live AI Stream">
        </div>
        <div class="stream-bar">
            <div class="tag-group">
                <span class="tag active">📷 GENERAL WEBCAM</span>
                <span class="tag" id="ram-tag">💾 RAM: Checking...</span>
            </div>
            <div class="tag-group">
                <span class="tag" id="fps-tag">⚡ 25 FPS</span>
                <span class="tag" id="status-tag">🟢 ONLINE</span>
            </div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <div>
                <div class="card-label">Hardware Device</div>
                <div class="card-val">Orange Pi Zero 3</div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Linux Ubuntu (aarch64) - ≤1GB RAM Target</div>
            </div>
            <div style="margin-top: 12px;">
                <div class="card-label">AI Inference Engine</div>
                <div class="card-val" style="color: #38BDF8;">{ai.backend_name.upper() if ai else "OFFLINE"}</div>
            </div>
        </div>

        <div class="card">
            <div>
                <div class="card-label">Real-time Waste Detector</div>
                <div style="margin: 8px 0 12px 0;">
                    <label style="font-size: 13px; color: #94A3B8; display: flex; justify-content: space-between;">
                        <span>🎚️ ปรับค่าความมั่นใจ (Confidence Threshold):</span>
                        <b id="conf-val" style="color: #10B981;">{int(ai.confidence_threshold * 100) if ai else 35}%</b>
                    </label>
                    <div class="confidence-meter" aria-label="Frozen confidence threshold"><span style="width: {int(ai.confidence_threshold * 100) if ai else 35}%"></span></div>
                    <div style="font-size: 11px; color: #64748B; margin-top: 7px;">ค่า threshold ถูก freeze สำหรับ demo เพื่อให้ผลคงที่</div>
                </div>
                <button class="btn-action" onclick="triggerInference()">⚡ ตรวจจับและวิเคราะห์ขยะ</button>
            </div>
            <div class="result-box" id="ai-result">
                💡 วางสิ่งของหน้ากล้องเพื่อดู <b>กรอบสีเหลี่ยม Bounding Box</b> และชนิดขยะแบบสดๆ
            </div>
        </div>
    </div>

    <script>
        let isAnnotated = true;
        const apiKeyParam = "{key_param}";
        const streamImg = document.getElementById('stream-img');
        const toggleBtn = document.getElementById('toggle-box-btn');

        function updateConf(val) {{
            document.getElementById('conf-val').innerText = val + '%';
        }}

        async function applyConf(val) {{
            const confFloat = parseFloat(val) / 100.0;
            try {{
                await fetch('/config/confidence{first_key_param}', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{confidence: confFloat}})
                }});
            }} catch(e) {{}}
        }}

        function toggleAnnotation() {{
            isAnnotated = !isAnnotated;
            const annotateVal = isAnnotated ? '1' : '0';
            streamImg.src = '/stream?annotate=' + annotateVal + apiKeyParam;
            if (isAnnotated) {{
                toggleBtn.innerText = '🎯 กล่อง AI: เปิด';
                toggleBtn.className = 'btn-toggle on';
            }} else {{
                toggleBtn.innerText = '🎯 กล่อง AI: ปิด';
                toggleBtn.className = 'btn-toggle';
            }}
        }}

        async function updateStats() {{
            try {{
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('ram-tag').innerText = '💾 RAM: ' + data.process_ram_mb + ' MB / ' + data.ram_limit_mb + ' MB';
            }} catch(e) {{}}
        }}
        setInterval(updateStats, 3000);
        updateStats();

        async function triggerInference() {{
            const resBox = document.getElementById('ai-result');
            resBox.innerHTML = '⏳ กำลังวิเคราะห์ภาพถ่าย...';
            try {{
                const res = await fetch('/capture{first_key_param}');
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || data.error || ('HTTP ' + res.status));
                const waste = data.class || 'unknown';
                const pillClass = 'pill-' + waste;
                const co2 = data.co2_offset_kg || 0.0;
                const conf = (Math.max(0, Math.min(1, Number(data.confidence || 0))) * 100).toFixed(1);
                const item = data.item_label || data.item_class || 'unknown';
                const thaiAliases = {{'Laptop':'แล็ปท็อป','Keyboard':'คีย์บอร์ด','mouse':'เมาส์','phone':'โทรศัพท์','Battery Charger':'ที่ชาร์จแบตเตอรี่','Cable Charger':'สายชาร์จ','adapter':'อะแดปเตอร์','laptopadapter':'อะแดปเตอร์แล็ปท็อป'}};
                const displayItem = thaiAliases[item] || item;
                
                let detectedText = '';
                if (data.detections && data.detections.length > 0) {{
                    detectedText = '<br><span style="color:#94A3B8; font-size:12px;">ตรวจพบ ' + data.detections.length + ' วัตถุในกรอบ</span>';
                }}

                resBox.innerHTML = '<div class="result-main">' +
                    '<span class="category-pill ' + pillClass + '">' + waste.toUpperCase() + '</span> ' +
                    '<div class="result-title">' + displayItem + '</div>' +
                    '<div class="result-subtitle">' + (data.model?.backend || 'edge') + ' • ' + (data.latency_ms || 0) + ' ms</div>' +
                    '<div class="result-confidence">' + conf + '%</div>' +
                    detectedText +
                    '<div style="margin-top: 6px; font-size: 12px; color: #34D399;">🌱 ลด CO2 ได้: -' + co2 + ' kg | ' + data.latency_ms + 'ms</div>' +
                    '</div>';
            }} catch(e) {{
                resBox.innerHTML = '❌ ผิดพลาด: ' + e;
            }}
        }}
    </script>
</body>
</html>"""
        return HTMLResponse(content=html)

    @app.post("/config/confidence")
    async def set_confidence_endpoint(request: Request):
        try:
            body = await request.json()
            conf = float(body.get("confidence", 0.35))
            if ai:
                ai.set_confidence(conf)
            return {"status": "updated", "confidence_threshold": conf}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/config/confidence")
    async def get_confidence_endpoint():
        conf = ai.confidence_threshold if ai else 0.35
        return {"confidence_threshold": conf}

    @app.get("/keys")
    async def get_keys():
        return {"keys": list_keys()}

    @app.post("/keys/generate")
    async def gen_key(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        label = body.get("label", "api-generated")
        scopes = body.get("scopes", ["predict", "camera", "heartbeat"])
        raw_key, meta = generate_key(label=label, scopes=scopes)
        return {
            "api_key": raw_key,
            "label": meta["label"],
            "scopes": meta["scopes"],
            "created_at": meta["created_at"],
            "warning": "Save this key -- it will NOT be shown again!",
        }

    @app.get("/system/ram")
    async def system_ram():
        try:
            rss = get_process_rss_mb()
            sys_mem = get_system_mem_mb()
            return {
                "process_rss_mb": round(rss, 1),
                "limit_mb": cfg.max_ram_mb,
                "usage_pct": round(rss / cfg.max_ram_mb * 100, 1),
                "system": sys_mem,
                "status": (
                    "ok" if rss < cfg.warn_ram_mb
                    else ("warn" if rss < cfg.max_ram_mb else "critical")
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/system/gc")
    async def trigger_gc():
        before = get_process_rss_mb()
        collected = force_gc()
        after = get_process_rss_mb()
        return {
            "before_mb": round(before, 1),
            "after_mb": round(after, 1),
            "freed_mb": round(max(before - after, 0), 1),
            "gc_collected": collected,
        }

    return app


# =============================================================================
# Entry point
# =============================================================================

def main():
    args = handle_cli()
    app = create_app(args)

    cfg = get_config()
    logger.info(
        f"Starting server on 0.0.0.0:{args.port} | "
        f"Camera: {'off' if args.no_camera else f'/dev/video{args.camera}'} | "
        f"RAM limit: {cfg.max_ram_mb}MB"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=1,                # Single worker = minimal RAM
        log_level="warning",      # Reduce uvicorn log noise on headless
        access_log=False,         # Reduce I/O on SD card / eMMC
        limit_concurrency=10,     # Bound concurrent connections
        limit_max_requests=50000, # Auto-restart to prevent leaks
        timeout_keep_alive=5,     # Short keep-alive for IoT clients
    )


if __name__ == "__main__":
    main()
