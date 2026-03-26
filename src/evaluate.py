from __future__ import annotations

from pathlib import Path


def run_dummy_eval(log_path: Path, *, accuracy: float, f1: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "[eval] starting synthetic evaluation",
                f"[eval] accuracy={accuracy:.3f} f1={f1:.3f}",
                "[eval] done",
                "",
            ]
        ),
        encoding="utf-8",
    )
