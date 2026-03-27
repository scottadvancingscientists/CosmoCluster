from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether a run contains evidence of real clustering and backend orchestration."
    )
    parser.add_argument("run_dir", help="Path to outputs/runs/<RUN_ID>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    train_log_path = run_dir / "logs/train.log"
    eval_log_path = run_dir / "logs/eval.log"
    orchestration_log_path = run_dir / "logs/orchestration.log"
    collect_metadata_path = run_dir / "collect_metadata.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    train_log = train_log_path.read_text(encoding="utf-8") if train_log_path.exists() else ""
    eval_log = eval_log_path.read_text(encoding="utf-8") if eval_log_path.exists() else ""
    orchestration_log = (
        orchestration_log_path.read_text(encoding="utf-8") if orchestration_log_path.exists() else ""
    )
    collect_metadata = (
        json.loads(collect_metadata_path.read_text(encoding="utf-8"))
        if collect_metadata_path.exists()
        else {}
    )

    checks: list[tuple[str, bool]] = [
        ("manifest.json exists", manifest_path.exists()),
        ("metrics.json exists", metrics_path.exists()),
        ("manifest.status == completed", manifest.get("status") == "completed"),
        ("train.log contains real clustering marker", "real synthetic clustering run" in train_log),
        ("train.log includes fit_seconds", "fit_seconds=" in train_log),
        ("eval.log contains purity/f1 marker", "purity=" in eval_log and "f1=" in eval_log),
        ("orchestration.log contains attempt record", "attempt=" in orchestration_log and "status=" in orchestration_log),
        (
            "metrics include accuracy/f1/loss/latency_ms",
            all(k in metrics for k in ("accuracy", "f1", "loss", "latency_ms")),
        ),
    ]

    backend = str(manifest.get("backend", "unknown"))
    if backend == "modal":
        checks.extend(
            [
                ("collect_metadata.json exists (modal)", collect_metadata_path.exists()),
                (
                    "collect_metadata.mode is credentialed or simulated",
                    collect_metadata.get("mode") in {"credentialed", "simulated"},
                ),
                ("collect_metadata has modal job_id", str(collect_metadata.get("job_id", "")).startswith("modal-")),
            ]
        )

    print(f"Run: {run_dir}")
    print(f"Backend: {backend}")
    print("")
    for label, flag in checks:
        print(f"[{_ok(flag)}] {label}")

    failures = [label for label, flag in checks if not flag]
    print("")
    if failures:
        print("Result: FAIL")
        for label in failures:
            print(f" - {label}")
        raise SystemExit(1)
    print("Result: PASS")


if __name__ == "__main__":
    main()
