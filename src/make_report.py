from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinja2 import Environment, FileSystemLoader


def generate_run_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary_md = (run_dir / "run_summary.md").read_text(encoding="utf-8")

    env = Environment(loader=FileSystemLoader("reports/templates"))
    template = env.get_template("run_report.html.j2")
    html = template.render(manifest=manifest, metrics=metrics, summary_md=summary_md)

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report = generate_run_report(Path(args.run_dir))
    print(report)


if __name__ == "__main__":
    main()
