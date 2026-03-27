from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from runners.base import Runner


class ModalRunner(Runner):
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _credential_status() -> dict[str, dict[str, Any]]:
        required = ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
        status: dict[str, dict[str, Any]] = {}
        for name in required:
            value = os.getenv(name)
            status[name] = {
                "present": value is not None,
                "non_empty": bool(value),
            }
        return status

    @staticmethod
    def _missing_credentials(status: dict[str, dict[str, Any]]) -> list[str]:
        missing: list[str] = []
        for key, details in status.items():
            if not details.get("present"):
                missing.append(f"{key} (missing env var; check GitHub secret mapping)")
            elif not details.get("non_empty"):
                missing.append(f"{key} (present but empty; check secret value)")
        return missing

    def submit(self, run_spec: Dict[str, Any]) -> str:
        run_id = run_spec["run_id"]
        job_id = f"modal-{run_id}"
        credential_status = self._credential_status()
        missing = self._missing_credentials(credential_status)
        self._jobs[job_id] = {
            "status": "submitted",
            "run_spec": run_spec,
            "credential_status": credential_status,
            "missing_credentials": missing,
            "submitted_at": time.perf_counter(),
        }
        return job_id

    def status(self, job_id: str) -> str:
        return self._jobs.get(job_id, {}).get("status", "unknown")

    def wait(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs[job_id]
        src_dir = Path(job["run_spec"]["local_output_dir"]).resolve()
        queue_timeout_seconds = int(job["run_spec"].get("queue_timeout_seconds", 900))
        requested_hardware = {
            "instance_type": str(job["run_spec"].get("compute_target", {}).get("instance_type", "unknown")),
            "accelerator": str(job["run_spec"].get("compute_target", {}).get("accelerator", "none")),
        }

        if not src_dir.exists():
            job["status"] = "failed"
            return {
                "status": "failed",
                "error": f"missing local output dir: {src_dir}",
                "modal_service_used": False,
                "requested_hardware": requested_hardware,
            }
        if queue_timeout_seconds <= 0:
            job["status"] = "failed"
            return {
                "status": "failed",
                "error": "queue timeout reached before run start",
                "modal_service_used": False,
                "requested_hardware": requested_hardware,
            }

        elapsed = time.perf_counter() - float(job.get("submitted_at", time.perf_counter()))
        missing = list(job.get("missing_credentials", []))
        has_credentials = len(missing) == 0

        # Explicit execution contract: this starter runner does not launch remote Modal compute.
        # It captures orchestration intent and credential state while preserving artifact contract.
        modal_service_used = False
        mode = "credentialed" if has_credentials else "simulated"
        notes = (
            "Modal credentials detected, but remote Modal execution is not yet implemented in this runner."
            if has_credentials
            else "Modal credentials missing or invalid; simulated orchestration fallback was used."
        )

        job["status"] = "completed"
        return {
            "status": "completed",
            "source_dir": str(src_dir),
            "modal_mode": mode,
            "modal_service_used": modal_service_used,
            "requested_hardware": requested_hardware,
            "runner_wait_seconds": round(elapsed, 4),
            "credential_status": job.get("credential_status", {}),
            "missing_credentials": missing,
            "notes": notes,
            "error": "" if has_credentials else "; ".join(missing),
        }

    def collect(self, job_id: str, output_dir: Path) -> Path:
        job = self._jobs[job_id]
        output_dir = output_dir.resolve()
        credential_status = job.get("credential_status", {})
        missing_credentials = job.get("missing_credentials", [])
        has_credentials = len(missing_credentials) == 0

        metadata = {
            "job_id": job_id,
            "backend": "modal",
            "collected": True,
            "mode": "credentialed" if has_credentials else "simulated",
            "modal_service_used": False,
            "requested_hardware": {
                "instance_type": str(job["run_spec"].get("compute_target", {}).get("instance_type", "unknown")),
                "accelerator": str(job["run_spec"].get("compute_target", {}).get("accelerator", "none")),
            },
            "notes": job["run_spec"].get("notes", ""),
            "attempt": int(job["run_spec"].get("attempt", 1)),
            "queue_timeout_seconds": int(job["run_spec"].get("queue_timeout_seconds", 900)),
            "credential_status": credential_status,
            "missing_credentials": missing_credentials,
            "duration_seconds": round(time.perf_counter() - float(job.get("submitted_at", time.perf_counter())), 4),
            "failure_hint": (
                "Map GitHub Actions secrets MODAL_TOKEN_ID and MODAL_TOKEN_SECRET into environment variables."
                if not has_credentials
                else "Remote Modal compute call is not yet implemented in this repository starter runner."
            ),
        }
        (output_dir / "collect_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        lines = [
            "[modal] runner completed",
            f"[modal] mode={metadata['mode']} modal_service_used={metadata['modal_service_used']}",
            (
                "[modal] requested_hardware="
                f"instance_type={metadata['requested_hardware']['instance_type']} "
                f"accelerator={metadata['requested_hardware']['accelerator']}"
            ),
            f"[modal] duration_seconds={metadata['duration_seconds']:.4f}",
        ]
        if metadata["missing_credentials"]:
            lines.append("[modal] missing_credentials=" + " | ".join(metadata["missing_credentials"]))
        lines.append(f"[modal] failure_hint={metadata['failure_hint']}")
        (output_dir / "logs/modal_runner.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_dir
