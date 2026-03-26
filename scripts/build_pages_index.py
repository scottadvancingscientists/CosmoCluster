from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    runs_root = Path("outputs/runs")
    site_root = Path("site")
    site_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(runs_root.glob("*"), reverse=True)[:20]:
        m = run_dir / "manifest.json"
        if not m.exists():
            continue
        manifest = json.loads(m.read_text(encoding="utf-8"))
        rows.append(manifest)

    html = [
        "<!doctype html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<style>body{font-family:-apple-system;padding:12px} .card{border:1px solid #ddd;border-radius:10px;padding:10px;margin:8px 0}</style>",
        "</head><body><h1>Recent runs</h1>",
    ]
    for r in rows:
        html.append(
            f"<div class='card'><div><b>{r['run_id']}</b> ({r['status']})</div><a href='runs/{r['run_id']}/report/index.html'>Open report</a></div>"
        )
    html.append("</body></html>")

    (site_root / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
