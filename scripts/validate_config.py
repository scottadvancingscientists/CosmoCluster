from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _require_runtime_dependencies() -> None:
    if importlib.util.find_spec("jsonschema") is None:
        raise SystemExit(
            "Missing dependency: jsonschema\n"
            "Install dependencies first:\n"
            "  pip install -r requirements.txt"
        )


def main() -> None:
    _require_runtime_dependencies()
    from src.utils.config import load_and_validate_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    cfg = load_and_validate_config(
        Path(args.config_path), Path("experiments/schemas/experiment_schema.yaml")
    )
    print(f"valid: {cfg['experiment_name']}")


if __name__ == "__main__":
    main()
