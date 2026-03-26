from __future__ import annotations

from pathlib import Path


def run_dummy_train(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "[train] starting dummy training\n[train] epoch=20 final_loss=0.221\n[train] done\n",
        encoding="utf-8",
    )
