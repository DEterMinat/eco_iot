"""
ECO-Gradian IoT Edge — API Key Manager.

Generates, validates, and revokes API keys so external devices/apps can
authenticate to this edge server.  Keys are persisted in a lightweight JSON file.
"""
import json
import secrets
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import KEYS_FILE, DATA_DIR

logger = logging.getLogger("eco_iot.key_manager")

# Key prefix makes tokens easily identifiable
_KEY_PREFIX = "ek_"
_HASH_ALGO = "sha256"


def _hash_key(raw_key: str) -> str:
    """One-way hash so we never store plaintext keys on disk."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _load_db() -> Dict:
    """Load key database from disk."""
    if KEYS_FILE.exists():
        try:
            return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"keys": {}, "revoked": []}


def _save_db(db: Dict):
    """Persist key database to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_key(label: str = "default", scopes: Optional[List[str]] = None) -> Tuple[str, dict]:
    """
    Generate a new API key.

    Returns:
        (raw_key, metadata) — raw_key is shown ONCE to the user; we store only hash.
    """
    raw_key = _KEY_PREFIX + secrets.token_hex(24)  # 48 hex chars = 192-bit entropy
    key_hash = _hash_key(raw_key)

    meta = {
        "label": label,
        "hash": key_hash,
        "scopes": scopes or ["predict", "camera", "heartbeat"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_used": None,
        "request_count": 0,
        "active": True,
    }

    db = _load_db()
    db["keys"][key_hash] = meta
    _save_db(db)

    logger.info(f"🔑 Generated API key for '{label}' (hash: {key_hash[:12]}...)")
    return raw_key, meta


def validate_key(raw_key: str) -> Optional[dict]:
    """
    Validate an incoming API key.

    Returns metadata dict if valid, None if rejected.
    """
    if not raw_key or not raw_key.startswith(_KEY_PREFIX):
        return None

    key_hash = _hash_key(raw_key)
    db = _load_db()

    if key_hash in db.get("revoked", []):
        logger.warning(f"🚫 Rejected revoked key (hash: {key_hash[:12]}...)")
        return None

    meta = db["keys"].get(key_hash)
    if meta is None:
        return None

    if not meta.get("active", True):
        logger.warning(f"🚫 Rejected inactive key '{meta.get('label')}'")
        return None

    # Update usage stats
    meta["last_used"] = datetime.now(timezone.utc).isoformat()
    meta["request_count"] = meta.get("request_count", 0) + 1
    _save_db(db)

    return meta


def revoke_key(raw_key: str) -> bool:
    """Revoke an API key permanently."""
    key_hash = _hash_key(raw_key)
    db = _load_db()

    if key_hash in db["keys"]:
        db["keys"][key_hash]["active"] = False
        if key_hash not in db.get("revoked", []):
            db.setdefault("revoked", []).append(key_hash)
        _save_db(db)
        label = db["keys"][key_hash].get("label", "?")
        logger.info(f"🚫 Revoked key '{label}' (hash: {key_hash[:12]}...)")
        return True

    return False


def list_keys() -> List[dict]:
    """List all registered keys (without exposing raw keys or full hashes)."""
    db = _load_db()
    result = []
    for h, meta in db["keys"].items():
        result.append({
            "key_id": h[:12] + "...",
            "label": meta.get("label", "?"),
            "scopes": meta.get("scopes", []),
            "active": meta.get("active", True),
            "created_at": meta.get("created_at"),
            "last_used": meta.get("last_used"),
            "request_count": meta.get("request_count", 0),
        })
    return result


def delete_key_by_prefix(prefix_12: str) -> bool:
    """Delete a key by its 12-char hash prefix (shown in list_keys)."""
    db = _load_db()
    target = None
    for h in db["keys"]:
        if h.startswith(prefix_12.rstrip(".")):
            target = h
            break
    if target:
        del db["keys"][target]
        _save_db(db)
        return True
    return False
