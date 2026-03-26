from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.gcp_batch_runner import GCPBatchRunner
from runners.local_runner import LocalRunner
from runners.modal_runner import ModalRunner
from src.evaluate import run_dummy_eval
from src.make_report import generate_run_report
from src.train import run_dummy_train
from src.utils.config import load_and_validate_config
from src.utils.io import ensure_dir, write_yaml
from src.utils.manifest import utc_now_iso, write_manifest
from src.utils.metrics import write_metrics
from src.utils.plotting import make_dummy_plots


def git_short_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
            .lower()
        )
    except Exception:
        return "nogit"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_run_id(experiment_name: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{git_short_sha()}_{experiment_name}_s{seed}"


def get_runner(name: str):
    if name == "local":
        return LocalRunner()
    if name == "modal":
        return ModalRunner()
    if name in {"gcp", "gcp_batch"}:
        return GCPBatchRunner()
    raise ValueError(f"Unsupported backend: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", default="local")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    schema_path = Path("experiments/schemas/experiment_schema.yaml")
    config = load_and_validate_config(config_path, schema_path)

    run_id = build_run_id(config["experiment_name"], int(config["seed"]))
    run_dir = Path("outputs/runs") / run_id
    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "plotly")
    ensure_dir(run_dir / "checkpoints")

    start = utc_now_iso()
    status = "completed"
    errors = []

    try:
        run_dummy_train(run_dir / "logs/train.log")
        run_dummy_eval(run_dir / "logs/eval.log")
        metrics = {"accuracy": 0.912, "f1": 0.901, "loss": 0.221, "latency_ms": 12.4}
        write_metrics(run_dir, metrics)
        make_dummy_plots(run_dir / "figures", run_dir / "plotly")
        (run_dir / "checkpoints/best.ckpt").write_text("dummy-checkpoint", encoding="utf-8")
        (run_dir / "run_summary.md").write_text(
            "# Run Summary\n\nDummy end-to-end run completed successfully.\n", encoding="utf-8"
        )
    except Exception as exc:
        status = "failed"
        errors.append(str(exc))
        trace_excerpt = traceback.format_exc(limit=8)
        (run_dir / "run_summary.md").write_text(
            f"# Failure Summary\n\nRun failed with error:\n\n```\n{trace_excerpt}\n```\n",
            encoding="utf-8",
        )
        if not (run_dir / "metrics.json").exists():
            write_metrics(run_dir, {"accuracy": 0.0, "f1": 0.0, "loss": 0.0, "latency_ms": 0.0})

    frozen_config_path = run_dir / "config_frozen.yaml"
    write_yaml(frozen_config_path, config)

    runner = get_runner(args.backend)
    run_spec = {"run_id": run_id, "local_output_dir": str(run_dir), "notes": args.notes}
    job_id = runner.submit(run_spec)
    wait_result = runner.wait(job_id)
    if wait_result.get("status") != "completed":
        status = "failed"
        errors.append(wait_result.get("error", "runner wait failed"))
    runner.collect(job_id, run_dir)

    end = utc_now_iso()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    manifest = {
        "run_id": run_id,
        "status": status,
        "start_time": start,
        "end_time": end,
        "duration": (end_dt - start_dt).total_seconds(),
        "git_sha": git_short_sha(),
        "branch": git_branch(),
        "config_path": str(config_path),
        "frozen_config_path": str(frozen_config_path),
        "artifact_paths": {
            "summary": "run_summary.md",
            "metrics_csv": "metrics.csv",
            "metrics_json": "metrics.json",
            "figures": "figures",
            "plotly": "plotly",
            "logs": "logs",
            "checkpoint": "checkpoints/best.ckpt",
        },
        "summary_metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "report_path": "report/index.html",
        "errors": errors,
    }
    write_manifest(run_dir / "manifest.json", manifest)
    report_path = generate_run_report(run_dir)
    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
