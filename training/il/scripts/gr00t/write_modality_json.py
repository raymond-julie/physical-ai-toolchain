"""Generate `meta/modality.json` for a converted LeRobot-v2.1 dataset.

GR00T's data loader requires a hand-authored `meta/modality.json` that declares
which flat-column slices of `observation.state` / `action` correspond to each
logical modality (arm, gripper) and which video features back each camera key.
The launcher cross-references this file with the `--modality-config-path`
Python module to assemble the input tensor.

Schema reference: demo_data/cube_to_bowl_5/meta/modality.json in Isaac-GR00T.

Example (Schaeffler bimanual UR5e, 12-D):

    python -m training.il.scripts.gr00t.write_modality_json \
        --dataset-dir /workspace/data/schaeffler_bimanual \
        --state-slices "right_arm=0:6,left_arm=6:12" \
        --action-slices "right_arm=0:6,left_arm=6:12" \
        --video "front=observation.images.d405_stationary_r_0" \
        --annotation "human.task_description=task_index"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def _parse_slices(spec: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for chunk in [c.strip() for c in spec.split(",") if c.strip()]:
        if "=" not in chunk or ":" not in chunk:
            raise ValueError(f"Invalid slice spec '{chunk}' (expected NAME=START:END)")
        name, span = chunk.split("=", 1)
        start_s, end_s = span.split(":", 1)
        start, end = int(start_s), int(end_s)
        if end <= start:
            raise ValueError(f"Slice {name} end ({end}) must be > start ({start})")
        out[name.strip()] = {"start": start, "end": end}
    return out


def _parse_mapping(spec: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for chunk in [c.strip() for c in spec.split(",") if c.strip()]:
        if "=" not in chunk:
            raise ValueError(f"Invalid mapping spec '{chunk}' (expected NAME=ORIGINAL_KEY)")
        name, original = chunk.split("=", 1)
        out[name.strip()] = {"original_key": original.strip()}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write meta/modality.json for a GR00T-ready LeRobot dataset.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Path to the converted LeRobot v2.1 dataset.")
    parser.add_argument("--state-slices", required=True, help="Comma-separated NAME=START:END entries for state.")
    parser.add_argument("--action-slices", required=True, help="Comma-separated NAME=START:END entries for action.")
    parser.add_argument("--video", required=True, help="Comma-separated NAME=ORIGINAL_KEY entries for video.")
    parser.add_argument(
        "--annotation",
        default="",
        help="Comma-separated NAME=ORIGINAL_KEY entries for annotation (optional).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing meta/modality.json.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[write_modality_json] %(message)s")

    meta_dir = args.dataset_dir / "meta"
    if not meta_dir.is_dir():
        _LOGGER.error("meta/ directory not found at %s; convert the dataset first.", meta_dir)
        return 2

    output = meta_dir / "modality.json"
    if output.exists() and not args.force:
        _LOGGER.info("%s already exists; skipping write (pass --force to overwrite).", output)
        return 0

    payload = {
        "state": _parse_slices(args.state_slices),
        "action": _parse_slices(args.action_slices),
        "video": _parse_mapping(args.video),
    }
    if args.annotation:
        payload["annotation"] = _parse_mapping(args.annotation)

    output.write_text(json.dumps(payload, indent=2) + "\n")
    _LOGGER.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
