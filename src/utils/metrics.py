from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from src.utils.io import write_json


def write_metrics(run_dir: Path, metrics: Dict[str, float]) -> None:
    write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
