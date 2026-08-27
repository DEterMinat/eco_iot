"""
ECO-Gradian IoT Edge — Memory Guard & Auto Cleanup Daemon.

Enforces hard RAM ceiling (default 900 MB), auto-prunes logs/cache,
and triggers Python GC on schedule.  Designed for 1 GB headless Linux SBCs.

Multi-layer RAM enforcement:
  Layer 1: Python GC + malloc_trim (every 30s)
  Layer 2: Application-level RSS check + auto-restart (every 60s)
  Layer 3: systemd MemoryMax=1024M (OS cgroup kill)
  Layer 4: Docker --memory=1g (container OOM kill)
"""
import gc
import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config, LOGS_DIR, DATA_DIR

logger = logging.getLogger("eco_iot.memory_guard")

# ── Set process-level soft/hard memory limit via rlimit (Linux only) ───────────
def _set_rlimit_rss(max_mb: int):
    """Set RLIMIT_AS (virtual address space) as a last-resort OS guard."""
    try:
        import resource
        max_bytes = max_mb * 1024 * 1024
        # Soft limit = warn, Hard limit = kill
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes + 128 * 1024 * 1024))
        logger.info(f"RLIMIT_AS set: soft={max_mb}MB hard={max_mb + 128}MB")
    except (ImportError, ValueError, OSError) as e:
        logger.debug(f"RLIMIT_AS not available: {e}")


def get_process_rss_mb() -> float:
    """Return current process RSS in megabytes.

    Reads /proc directly first (most reliable on headless Linux),
    then falls back to resource module, then psutil.
    """
    # 1. /proc/<pid>/status — always available on Linux IoT
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except (FileNotFoundError, OSError):
        pass

    # 2. resource module (Unix)
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024  # Linux: KB
    except (ImportError, OSError):
        pass

    # 3. psutil fallback
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        pass

    return 0.0


def get_system_mem_mb() -> dict:
    """Read total/available system RAM from /proc/meminfo (headless Linux)."""
    info = {"total_mb": 0, "available_mb": 0, "used_pct": 0}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["total_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable:"):
                    info["available_mb"] = int(line.split()[1]) / 1024
        if info["total_mb"] > 0:
            info["used_pct"] = round((1 - info["available_mb"] / info["total_mb"]) * 100, 1)
    except (FileNotFoundError, OSError):
        pass
    return info


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of directory in MB."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 * 1024)


def cleanup_logs(cfg=None):
    """Delete old log files and rotate if total size exceeds limit."""
    cfg = cfg or get_config()
    now = datetime.now()
    cutoff = now - timedelta(hours=cfg.log_max_age_hours)
    deleted = 0

    if not LOGS_DIR.exists():
        return

    try:
        for log_file in sorted(LOGS_DIR.glob("*.log*"), key=lambda p: p.stat().st_mtime):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff:
                    log_file.unlink()
                    deleted += 1
            except OSError:
                pass
    except OSError:
        pass

    # Size-based prune
    try:
        while _dir_size_mb(LOGS_DIR) > cfg.log_max_size_mb:
            oldest = sorted(LOGS_DIR.glob("*.log*"), key=lambda p: p.stat().st_mtime)
            if not oldest:
                break
            try:
                oldest[0].unlink()
                deleted += 1
            except OSError:
                break
    except OSError:
        pass

    if deleted:
        logger.info(f"Cleanup: removed {deleted} log files")


def cleanup_cache(cfg=None):
    """Aggressively auto-clear ALL temp, cache, bytecode, and stale files.

    IoT devices have limited flash/eMMC — don't wait for budget overflow.
    This runs every cycle unconditionally.
    """
    cfg = cfg or get_config()
    cleaned = 0

    # 1. Inference results cache (data/cache/) — keep only newest files under budget
    cache_dir = DATA_DIR / "cache"
    if cache_dir.exists():
        try:
            files = sorted(cache_dir.rglob("*"), key=lambda p: p.stat().st_mtime)
            # Always delete files older than 1 hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            for f in files:
                if f.is_file():
                    try:
                        if datetime.fromtimestamp(f.stat().st_mtime) < one_hour_ago:
                            f.unlink()
                            cleaned += 1
                    except OSError:
                        pass

            # Then size-based prune for remaining files
            while _dir_size_mb(cache_dir) > cfg.cache_max_size_mb:
                remaining = sorted(cache_dir.rglob("*"), key=lambda p: p.stat().st_mtime)
                remaining = [f for f in remaining if f.is_file()]
                if not remaining:
                    break
                try:
                    remaining[0].unlink()
                    cleaned += 1
                except OSError:
                    break
        except OSError:
            pass

    # 2. __pycache__ / *.pyc / *.pyo bytecode — useless on production IoT
    import shutil
    project_root = Path(__file__).parent
    for pycache in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
            cleaned += 1
        except OSError:
            pass
    for pyc in project_root.rglob("*.pyc"):
        try:
            pyc.unlink()
            cleaned += 1
        except OSError:
            pass
    for pyo in project_root.rglob("*.pyo"):
        try:
            pyo.unlink()
            cleaned += 1
        except OSError:
            pass

    # 3. /tmp/eco_iot and stale temp files
    for tmp_path in [Path("/tmp/eco_iot"), Path("/tmp/opencv"), Path("/tmp/tflite")]:
        try:
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
                cleaned += 1
        except OSError:
            pass

    # 4. Stale .tmp / .bak files in data/
    try:
        for pattern in ("*.tmp", "*.bak", "*.old", "*.swp"):
            for f in DATA_DIR.rglob(pattern):
                try:
                    f.unlink()
                    cleaned += 1
                except OSError:
                    pass
    except OSError:
        pass

    # 5. Truncate journal/syslog buffer if accessible (Docker/systemd)
    try:
        import subprocess
        subprocess.run(
            ["journalctl", "--vacuum-size=5M"],
            capture_output=True, timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    # 6. Drop Linux page cache (requires root or Docker --privileged)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1")  # Free pagecache
    except (FileNotFoundError, PermissionError, OSError):
        pass

    if cleaned:
        logger.info(f"Auto-cleanup: cleared {cleaned} stale files")


def force_gc() -> int:
    """Aggressive garbage collection + malloc_trim for headless Linux."""
    # Disable GC thresholds temporarily for thorough sweep
    old_thresholds = gc.get_threshold()
    gc.set_threshold(0, 0, 0)

    collected = gc.collect(generation=2)
    gc.collect(generation=1)
    gc.collect(generation=0)

    gc.set_threshold(*old_thresholds)

    # Release malloc arenas back to OS (Linux glibc)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass

    return collected


def check_ram(cfg=None) -> float:
    """Check current RAM usage and take action if over thresholds.

    Actions (escalating):
      1. warn_ram_mb exceeded -> force GC
      2. max_ram_mb exceeded -> GC + cache purge + log purge
      3. Still over after cleanup -> self-restart via os.execv
    """
    cfg = cfg or get_config()
    rss = get_process_rss_mb()

    if rss > cfg.max_ram_mb:
        logger.critical(
            f"RAM CRITICAL: {rss:.0f} MB > {cfg.max_ram_mb} MB ceiling! "
            f"Force GC + full purge."
        )
        force_gc()
        cleanup_cache(cfg)
        cleanup_logs(cfg)

        # Re-check after cleanup
        rss = get_process_rss_mb()
        if rss > cfg.max_ram_mb:
            logger.critical(f"RAM still {rss:.0f} MB after cleanup. Restarting process...")
            # Flush logs before restart
            logging.shutdown()
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except OSError:
                # Last resort: SIGTERM self -> Docker/systemd will restart us
                os.kill(os.getpid(), signal.SIGTERM)

    elif rss > cfg.warn_ram_mb:
        logger.warning(f"RAM HIGH: {rss:.0f} MB (warn: {cfg.warn_ram_mb} MB)")
        force_gc()

    return rss


class MemoryGuardDaemon(threading.Thread):
    """Background daemon that periodically checks RAM, runs GC, and cleans logs/cache.

    Crash-resilient: all operations wrapped in try/except so a monitoring
    failure never takes down the main server process.
    """

    def __init__(self, cfg=None):
        super().__init__(daemon=True, name="MemoryGuard")
        self.cfg = cfg or get_config()
        self._stop_event = threading.Event()
        self._gc_tick = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        cfg = self.cfg
        logger.info(
            f"MemoryGuard started | ceiling={cfg.max_ram_mb}MB "
            f"gc_every={cfg.gc_interval_sec}s cleanup_every={cfg.cleanup_interval_sec}s"
        )

        # Set OS-level memory guard on Linux
        _set_rlimit_rss(cfg.max_ram_mb + 200)

        gc_interval = cfg.gc_interval_sec
        cleanup_interval = cfg.cleanup_interval_sec

        while not self._stop_event.is_set():
            try:
                self._gc_tick += gc_interval

                # Always: RAM check + GC
                rss = check_ram(cfg)
                collected = force_gc()

                # Log system-level memory on headless Linux
                sys_mem = get_system_mem_mb()
                if sys_mem["total_mb"] > 0:
                    logger.debug(
                        f"RAM: process={rss:.0f}MB | "
                        f"system={sys_mem['used_pct']}% used "
                        f"({sys_mem['available_mb']:.0f}MB free / {sys_mem['total_mb']:.0f}MB total) | "
                        f"gc={collected}"
                    )
                else:
                    logger.debug(f"RAM: {rss:.0f}MB | gc={collected}")

                # Periodic: Log + Cache cleanup
                if self._gc_tick >= cleanup_interval:
                    cleanup_logs(cfg)
                    cleanup_cache(cfg)
                    self._gc_tick = 0

            except Exception as e:
                # Guard thread must never crash
                logger.error(f"MemoryGuard tick error (non-fatal): {e}")

            self._stop_event.wait(gc_interval)

        logger.info("MemoryGuard stopped.")
