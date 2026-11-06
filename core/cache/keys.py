"""Content-addressed stage cache keys (SPEC §7.1, D11)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def stage_key(video_hash: str, stage_name: str, config: dict, upstream_key: str = "") -> str:
    """Chained hash: changing an upstream stage invalidates every stage downstream of it."""
    payload = "|".join((video_hash, stage_name, canonical_json(config), upstream_key))
    return hashlib.sha256(payload.encode()).hexdigest()
