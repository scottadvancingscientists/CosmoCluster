from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jsonschema import validate

from src.utils.io import read_yaml


def load_and_validate_config(config_path: Path, schema_path: Path) -> Dict[str, Any]:
    config = read_yaml(config_path)
    schema = read_yaml(schema_path)
    validate(instance=config, schema=schema)
    return config
