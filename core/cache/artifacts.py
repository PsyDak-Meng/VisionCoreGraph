"""Stage artifact read/write, keyed by cache.keys.stage_key.

v0 stores every stage as JSON rather than the spec's Parquet/JSONL split —
simplest thing that gives real resumability without adding a pandas/pyarrow
dependency. Revisit if artifact size becomes a problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def artifact_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def exists(cache_dir: Path, key: str) -> bool:
    return artifact_path(cache_dir, key).exists()


def save(cache_dir: Path, key: str, data: Any) -> None:
    path = artifact_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def load(cache_dir: Path, key: str) -> Any:
    return json.loads(artifact_path(cache_dir, key).read_text())
