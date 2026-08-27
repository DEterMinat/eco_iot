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

# ── FastAPI imports ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException, Response
    from fastapi.responses import JSONResponse
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
                "camera_open": camera.is_open if camera else False,
                "frames_captured": camera.frame_count if camera else 0,
                "uptime_sec": round(time.process_time(), 1),
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
