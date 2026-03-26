from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class Runner(ABC):
    @abstractmethod
    def submit(self, run_spec: Dict[str, Any]) -> str:
        ...

    @abstractmethod
    def status(self, job_id: str) -> str:
        ...

    @abstractmethod
    def wait(self, job_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def collect(self, job_id: str, output_dir: Path) -> Path:
        ...
