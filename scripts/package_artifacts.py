from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--out", default="outputs/packages")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_base = out_dir / run_dir.name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    print(archive)


if __name__ == "__main__":
    main()
