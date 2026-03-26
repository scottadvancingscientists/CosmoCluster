from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from runners.base import Runner


class GCPBatchRunner(Runner):
    def submit(self, run_spec: Dict[str, Any]) -> str:
        raise NotImplementedError("GCP Batch runner will be implemented in a later phase.")

    def status(self, job_id: str) -> str:
        return "not_implemented"

    def wait(self, job_id: str) -> Dict[str, Any]:
        raise NotImplementedError("GCP Batch runner will be implemented in a later phase.")

    def collect(self, job_id: str, output_dir: Path) -> Path:
        raise NotImplementedError("GCP Batch runner will be implemented in a later phase.")
