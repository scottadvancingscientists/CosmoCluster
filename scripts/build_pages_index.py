from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.artifacts_index import write_artifacts_index


def _load_metrics(run_dir: Path) -> dict[str, str]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    out: dict[str, str] = {}
    for key in ("accuracy", "f1", "loss"):
        value = data.get(key)
        if isinstance(value, int | float):
            out[key] = f"{value:.4f}"
    return out


def _fmt_created_at(iso_ts: str) -> str:
    if not iso_ts:
        return "unknown"
    try:
        parsed = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return iso_ts
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    runs_root = Path("outputs/runs")
    site_root = Path("site")
    site_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(runs_root.glob("*"), reverse=True)[:20]:
        m = run_dir / "manifest.json"
        if not m.exists():
            continue
        if not (run_dir / "artifacts/index.html").exists():
            write_artifacts_index(run_dir)
        manifest = json.loads(m.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": manifest.get("run_id", run_dir.name),
                "status": manifest.get("status", "unknown"),
                "backend": manifest.get("backend", "unknown"),
                "created_at": _fmt_created_at(manifest.get("created_at", "")),
                "metrics": _load_metrics(run_dir),
            }
        )

    html = [
        "<!doctype html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>CosmoCluster runs</title>",
        "<link rel='stylesheet' href='../reports/static/styles.css'>",
        "</head><body>",
        "<main>",
        "<section class='card'>",
        "<h1 style='margin:.2rem 0'>Recent runs</h1>",
        "<p class='muted' style='margin:0'>Phone-first dashboard for latest experiment artifacts.</p>",
        (
            f"<p style='margin:.75rem 0 0 0'><a class='button-link' href='runs/{rows[0]['run_id']}/report/index.html'>"
            "Open latest report</a></p>"
            if rows
            else ""
        ),
        "</section>",
    ]

    if not rows:
        html.append(
            "<section class='card'><p>No runs found yet. Trigger <code>run-experiment</code> from GitHub Actions.</p></section>"
        )

    for r in rows:
        metric_html = " ".join(
            f"<span class='badge metric-badge'>{k}: {v}</span>" for k, v in r["metrics"].items()
        ) or "<span class='muted'>No metrics captured</span>"
        html.append(
            "".join(
                [
                    "<section class='card'>",
                    f"<div><strong>{r['run_id']}</strong></div>",
                    f"<div class='badge {'ok' if r['status'] == 'completed' else 'bad'}'>{r['status']}</div>",
                    f"<p class='muted' style='margin:.5rem 0'>Backend: {r['backend']} · {r['created_at']}</p>",
                    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{metric_html}</div>",
                    (
                        "<p style='margin:.75rem 0 0 0;display:flex;gap:8px;flex-wrap:wrap'>"
                        f"<a class='button-link' href='runs/{r['run_id']}/report/index.html'>Open report</a>"
                        f"<a class='button-link secondary' href='runs/{r['run_id']}/artifacts/index.html'>Browse artifacts</a>"
                        f"<a class='button-link secondary' href='runs/{r['run_id']}/artifacts.zip'>Download .zip</a>"
                        "</p>"
                    ),
                    "</section>",
                ]
            )
        )

    html.extend(["</main>", "</body></html>"])

    (site_root / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
