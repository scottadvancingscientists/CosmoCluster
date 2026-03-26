from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinja2 import Environment, FileSystemLoader


def compare(run_dirs: list[Path]) -> Path:
    compare_id = datetime.now(timezone.utc).strftime("cmp_%Y%m%d_%H%M%S")
    out_dir = Path("outputs/comparisons") / compare_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = None
    best_acc = -1.0
    for rd in run_dirs:
        manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        row = {"run_id": manifest["run_id"], "status": manifest["status"], **metrics}
        rows.append(row)
        if metrics.get("accuracy", -1) > best_acc:
            best_acc = metrics["accuracy"]
            best = manifest["run_id"]

    env = Environment(loader=FileSystemLoader("reports/templates"))
    template = env.get_template("compare_report.html.j2")
    html = template.render(compare_id=compare_id, rows=rows, best_run_id=best)
    output = out_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    args = parser.parse_args()
    out = compare([Path(p) for p in args.run_dirs])
    print(out)


if __name__ == "__main__":
    main()
