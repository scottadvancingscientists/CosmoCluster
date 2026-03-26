from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.io import write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
