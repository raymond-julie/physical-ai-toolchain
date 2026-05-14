"""Thin wrapper around Isaac-GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py`.

Runs the upstream converter against a LeRobot v3 dataset directory and produces
an in-place v2.1 layout (the original is renamed to `<dataset>_v30` by the
upstream script). Exposed as a module so the AzureML entry script invokes it
identically to the OFT helpers:

    python -m training.il.scripts.gr00t.lerobot_v3_to_v2 \
        --gr00t-dir /workspace/Isaac-GR00T \
        --dataset-dir /workspace/data/schaeffler_bimanual \
        --repo-id schaeffler_bimanual

The wrapper:

* Resolves the upstream converter path from `--gr00t-dir`.
* Detects whether the dataset is already v2.1 (`meta/info.json` codebase_version
  starts with "v2") and exits with `0` if so, making it safe to re-run.
* Forwards remaining CLI flags to the upstream converter.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def _info_codebase_version(dataset_dir: Path) -> str | None:
    info = dataset_dir / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        return json.loads(info.read_text()).get("codebase_version")
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a LeRobot v3 dataset to v2.1 via Isaac-GR00T's converter.")
    parser.add_argument("--gr00t-dir", required=True, type=Path, help="Path to the Isaac-GR00T clone.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Path to the LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="Logical repo id (matches the dataset folder name).")
    parser.add_argument("--force", action="store_true", help="Force conversion even if v2.1 layout is detected.")
    args, forward = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="[lerobot_v3_to_v2] %(message)s")

    converter = args.gr00t_dir / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"
    if not converter.is_file():
        _LOGGER.error("Converter not found at %s", converter)
        return 2

    if not args.dataset_dir.is_dir():
        _LOGGER.error("Dataset directory not found at %s", args.dataset_dir)
        return 2

    codebase_version = _info_codebase_version(args.dataset_dir)
    if codebase_version and codebase_version.startswith("v2") and not args.force:
        _LOGGER.info("Dataset already at %s; skipping conversion (pass --force to override).", codebase_version)
        return 0

    cmd = [
        sys.executable,
        str(converter),
        "--repo-id",
        args.repo_id,
        "--root",
        str(args.dataset_dir.parent),
        *forward,
    ]
    _LOGGER.info("Running %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
