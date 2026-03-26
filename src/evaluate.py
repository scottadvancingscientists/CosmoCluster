from __future__ import annotations

from pathlib import Path


def run_dummy_eval(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "[eval] starting dummy evaluation\n[eval] accuracy=0.912 f1=0.901\n[eval] done\n",
        encoding="utf-8",
    )
