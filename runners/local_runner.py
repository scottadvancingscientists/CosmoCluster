from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from runners.base import Runner


class LocalRunner(Runner):
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, run_spec: Dict[str, Any]) -> str:
        job_id = run_spec["run_id"]
        self._jobs[job_id] = {"status": "submitted", "run_spec": run_spec}
        return job_id

    def status(self, job_id: str) -> str:
        return self._jobs.get(job_id, {}).get("status", "unknown")

    def wait(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs[job_id]
        src_dir = Path(job["run_spec"]["local_output_dir"]).resolve()
        if not src_dir.exists():
            job["status"] = "failed"
            return {"status": "failed", "error": f"missing local output dir: {src_dir}"}
        job["status"] = "completed"
        return {"status": "completed", "source_dir": str(src_dir)}

    def collect(self, job_id: str, output_dir: Path) -> Path:
        job = self._jobs[job_id]
        src_dir = Path(job["run_spec"]["local_output_dir"]).resolve()
        output_dir = output_dir.resolve()
        if src_dir != output_dir:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(src_dir, output_dir)
        (output_dir / "collect_metadata.json").write_text(
            json.dumps({"job_id": job_id, "collected": True}, indent=2), encoding="utf-8"
        )
        return output_dir
