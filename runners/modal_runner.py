from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from runners.base import Runner


class ModalRunner(Runner):
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, run_spec: Dict[str, Any]) -> str:
        run_id = run_spec["run_id"]
        job_id = f"modal-{run_id}"
        token_id = os.getenv("MODAL_TOKEN_ID")
        token_secret = os.getenv("MODAL_TOKEN_SECRET")
        self._jobs[job_id] = {
            "status": "submitted",
            "run_spec": run_spec,
            "has_modal_credentials": bool(token_id and token_secret),
        }
        return job_id

    def status(self, job_id: str) -> str:
        return self._jobs.get(job_id, {}).get("status", "unknown")

    def wait(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs[job_id]
        src_dir = Path(job["run_spec"]["local_output_dir"]).resolve()
        if not src_dir.exists():
            job["status"] = "failed"
            return {"status": "failed", "error": f"missing local output dir: {src_dir}"}

        # Phase-3 starter behavior:
        # We preserve the run artifact contract now and capture Modal metadata.
        # A later change can replace this with true remote execution/collection.
        job["status"] = "completed"
        mode = "credentialed" if job["has_modal_credentials"] else "simulated"
        return {"status": "completed", "source_dir": str(src_dir), "modal_mode": mode}

    def collect(self, job_id: str, output_dir: Path) -> Path:
        job = self._jobs[job_id]
        output_dir = output_dir.resolve()
        metadata = {
            "job_id": job_id,
            "backend": "modal",
            "collected": True,
            "mode": "credentialed" if job["has_modal_credentials"] else "simulated",
            "notes": job["run_spec"].get("notes", ""),
        }
        (output_dir / "collect_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (output_dir / "logs/modal_runner.log").write_text(
            "ModalRunner completed in "
            f"{metadata['mode']} mode. "
            "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET for credentialed mode.\n",
            encoding="utf-8",
        )
        return output_dir
