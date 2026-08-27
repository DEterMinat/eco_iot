"""
ECO-Gradian IoT Edge — Test Suite.

Validates key_manager, memory_guard, ai_engine, and camera modules
without requiring actual hardware.
"""
import os
import sys
import json
import time
import tempfile
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── 1. Config Tests ────────────────────────────────────────────────────────────

def test_config_defaults():
    from config import get_config
    cfg = get_config()
    assert cfg.port == 8080
    assert cfg.max_ram_mb == 900
    assert cfg.camera_index == 0
    assert cfg.camera_width == 640
    assert cfg.camera_height == 480
    assert cfg.jpeg_quality == 80
    assert cfg.frame_buffer_max == 3
    assert cfg.log_max_size_mb == 20
    assert cfg.log_max_age_hours == 48
    assert "plastic" in cfg.class_labels
    assert cfg.carbon_factors["metal"] == 9.0
    print("  [OK] config defaults")


# ── 2. Key Manager Tests ──────────────────────────────────────────────────────

def test_key_lifecycle():
    from key_manager import generate_key, validate_key, list_keys, revoke_key

    # Generate
    raw_key, meta = generate_key(label="test-key-unit")
    assert raw_key.startswith("ek_")
    assert meta["label"] == "test-key-unit"
    assert meta["active"] is True
    print(f"  [OK] generate_key (key: {raw_key[:16]}...)")

    # Validate
    result = validate_key(raw_key)
    assert result is not None
    assert result["label"] == "test-key-unit"
    assert result["request_count"] >= 1
    print("  [OK] validate_key")

    # Invalid key rejected
    assert validate_key("ek_invalid_garbage") is None
    assert validate_key("") is None
    assert validate_key("not_a_key") is None
    print("  [OK] invalid keys rejected")

    # List
    keys = list_keys()
    assert any(k["label"] == "test-key-unit" for k in keys)
    print(f"  [OK] list_keys ({len(keys)} keys)")

    # Revoke
    assert revoke_key(raw_key) is True
    assert validate_key(raw_key) is None
    print("  [OK] revoke_key")


# ── 3. Memory Guard Tests ─────────────────────────────────────────────────────

def test_memory_guard():
    from memory_guard import get_process_rss_mb, force_gc, check_ram, get_system_mem_mb

    rss = get_process_rss_mb()
    assert rss >= 0
    print(f"  [OK] get_process_rss_mb ({rss:.1f} MB)")

    collected = force_gc()
    assert collected >= 0
    print(f"  [OK] force_gc (collected {collected} objects)")

    rss = check_ram()
    assert rss >= 0
    print(f"  [OK] check_ram ({rss:.1f} MB)")

    sys_mem = get_system_mem_mb()
    assert isinstance(sys_mem, dict)
    assert "total_mb" in sys_mem
    print(f"  [OK] get_system_mem_mb (total={sys_mem['total_mb']:.0f}MB avail={sys_mem['available_mb']:.0f}MB)")


# ── 4. AI Engine Tests ────────────────────────────────────────────────────────

def test_ai_engine():
    from ai_engine import AIEngine

    engine = AIEngine()
    assert engine.backend_name in ("tflite", "onnx", "heuristic", "contour_ai")
    print(f"  [OK] AIEngine loaded (backend: {engine.backend_name})")

    # Predict with synthetic JPEG bytes
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 500 + b"\xff\xd9"
    result = engine.predict(fake_jpeg)
    assert "waste_type" in result
    assert "confidence" in result
    assert "latency_ms" in result
    assert "co2_offset_kg" in result
    assert "disposal_action" in result
    assert result["confidence"] >= 0
    print(f"  [OK] predict: {result['waste_type']} ({result['confidence']*100:.0f}%) in {result['latency_ms']:.1f}ms")


# ── 5. Camera Tests (simulation) ──────────────────────────────────────────────

def test_camera_simulation():
    from camera import CameraManager

    cam = CameraManager(index=99, buffer_size=2)  # Index 99 = will fail, use simulation
    # Should still provide synthetic frames
    b64 = cam.grab_frame_base64()
    # In simulation mode with no open camera, may return None or synthetic
    jpeg = cam.grab_frame_jpeg()
    # Synthetic JPEG should be valid
    synthetic = cam._synthetic_jpeg()
    assert synthetic[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert synthetic[-2:] == b"\xff\xd9"  # JPEG EOI marker
    print(f"  [OK] Camera simulation (synthetic JPEG: {len(synthetic)} bytes)")


# ── Runner ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  ECO-Gradian IoT Edge -- Test Suite")
    print("=" * 50 + "\n")

    tests = [
        ("Config Defaults", test_config_defaults),
        ("Key Manager Lifecycle", test_key_lifecycle),
        ("Memory Guard", test_memory_guard),
        ("AI Engine", test_ai_engine),
        ("Camera Simulation", test_camera_simulation),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"\n[TEST] {name}:")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
