from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.gcp_batch_runner import GCPBatchRunner
from runners.local_runner import LocalRunner
from runners.modal_runner import ModalRunner
from src.make_report import generate_run_report
from src.real_experiment import run_real_experiment
from src.utils.artifacts_index import write_artifacts_index
from src.utils.config import load_and_validate_config
from src.utils.io import ensure_dir, write_yaml
from src.utils.manifest import utc_now_iso, write_manifest
from src.utils.metrics import write_metrics
from src.utils.plotting import make_run_plots


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
    parser.add_argument("--backend", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=int, default=10)
    parser.add_argument("--queue-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    config_path = Path(args.config)
    schema_path = Path("experiments/schemas/experiment_schema.yaml")
    config = load_and_validate_config(config_path, schema_path)
    if args.seed_override is not None:
        config["seed"] = int(args.seed_override)

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
        real_result = run_real_experiment(config, run_dir)
        metrics = real_result.metrics
        write_metrics(run_dir, metrics)
        make_run_plots(
            run_dir / "figures",
            run_dir / "plotly",
            case_name=real_result.case_name,
            y_true=real_result.y_true,
            y_pred=real_result.y_pred,
            epochs=real_result.epochs,
        )
        (run_dir / "checkpoints/best.ckpt").write_text("cosmocluster-minimal-checkpoint", encoding="utf-8")
        (run_dir / "run_summary.md").write_text(
            "\n".join(
                [
                    "# Run Summary",
                    "",
                    "Real synthetic clustering run completed successfully.",
                    "",
                    f"- case: {real_result.case_name}",
                    f"- fit_seconds: {real_result.fit_seconds:.4f}",
                    f"- accuracy: {metrics['accuracy']}",
                    f"- f1: {metrics['f1']}",
                    f"- loss: {metrics['loss']}",
                    f"- latency_ms: {metrics['latency_ms']}",
                    f"- backend: {config.get('compute_target', {}).get('backend', 'unknown')}",
                    "",
                ]
            ),
            encoding="utf-8",
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

    backend = args.backend or config.get("compute_target", {}).get("backend", "local")
    runner = get_runner(backend)
    run_spec = {
        "run_id": run_id,
        "local_output_dir": str(run_dir),
        "notes": args.notes,
        "backend": backend,
        "compute_target": config.get("compute_target", {}),
        "queue_timeout_seconds": max(0, int(args.queue_timeout_seconds)),
    }
    orchestration_log_path = run_dir / "logs/orchestration.log"
    max_retries = max(0, int(args.max_retries))
    retry_delay_seconds = max(0, int(args.retry_delay_seconds))
    attempt = 0
    wait_result: dict[str, object] = {"status": "failed", "error": "runner was never invoked"}
    job_id = ""

    while attempt <= max_retries:
        attempt += 1
        job_id = runner.submit({**run_spec, "attempt": attempt})
        wait_result = runner.wait(job_id)
        attempt_status = wait_result.get("status", "unknown")
        attempt_error = wait_result.get("error", "")
        modal_mode = wait_result.get("modal_mode", "")
        modal_service_used = wait_result.get("modal_service_used", "")
        requested_hardware = wait_result.get("requested_hardware", {})
        runner_wait_seconds = wait_result.get("runner_wait_seconds", "")
        missing_credentials = wait_result.get("missing_credentials", [])
        with orchestration_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                (
                    f"attempt={attempt} job_id={job_id} status={attempt_status} "
                    f"modal_mode={modal_mode} modal_service_used={modal_service_used} "
                    f"instance_type={requested_hardware.get('instance_type', 'unknown')} "
                    f"accelerator={requested_hardware.get('accelerator', 'none')} "
                    f"runner_wait_seconds={runner_wait_seconds} "
                    f"missing_credentials={'|'.join(missing_credentials)} "
                    f"error={attempt_error}\n"
                )
            )
        if attempt_status == "completed":
            break
        if attempt <= max_retries and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)

    if wait_result.get("status") != "completed":
        status = "failed"
        errors.append(wait_result.get("error", "runner wait failed"))
    if job_id:
        runner.collect(job_id, run_dir)

    end = utc_now_iso()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    manifest = {
        "run_id": run_id,
        "status": status,
        "backend": backend,
        "created_at": start,
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
            "case_metrics_csv": "case_metrics.csv",
            "comparator_metrics_csv": "comparator_metrics.csv",
            "comparator_metrics_json": "comparator_metrics.json",
            "figures": "figures",
            "plotly": "plotly",
            "logs": "logs",
            "checkpoint": "checkpoints/best.ckpt",
        },
        "summary_metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "report_path": "report/index.html",
        "orchestration": {
            "attempts": attempt,
            "max_retries": max_retries,
            "retry_delay_seconds": retry_delay_seconds,
            "queue_timeout_seconds": run_spec["queue_timeout_seconds"],
            "last_wait_result": wait_result,
        },
        "errors": errors,
    }
    write_manifest(run_dir / "manifest.json", manifest)
    report_path = generate_run_report(run_dir)
    artifacts_index_path = write_artifacts_index(run_dir)
    archive_path = shutil.make_archive(str(run_dir / "artifacts"), "zip", root_dir=run_dir)
    if archive_path:
        print(f"ARCHIVE={archive_path}")
    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"REPORT={report_path}")
    print(f"ARTIFACTS_INDEX={artifacts_index_path}")


if __name__ == "__main__":
    main()
