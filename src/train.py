from __future__ import annotations

from pathlib import Path


def run_dummy_train(log_path: Path, *, epochs: int, final_loss: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "[train] starting synthetic training",
                f"[train] epochs={epochs}",
                f"[train] epoch={epochs} final_loss={final_loss:.3f}",
                "[train] done",
                "",
            ]
        ),
        encoding="utf-8",
    )
