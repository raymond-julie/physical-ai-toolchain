"""Filter a LeRobot v3 dataset to complete + (optionally) VLM-judged-success episodes.

Produces a manifest JSON listing eligible episodes with per-episode metadata
(length, language instruction, video paths, success). Downstream consumer:
`lerobot_to_rlds.py`.

Usage:
    python -m training.il.scripts.openvla_oft.filter_dataset \
        --dataset datasets/schaeffler_sim_avc1/second_collection \
        --vlm-judge outputs/dataset-analysis/schaeffler_second_collection/vlm-judge.jsonl \
        --image-keys observation.images.d405_stationary_r_0 \
                     observation.images.d405_stationary_l_1 \
                     observation.images.d405_stationary_l_2 \
        --output datasets/schaeffler_sim_avc1/second_collection/training_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int
    language_instruction: str
    success: bool | None
    data_path: str
    video_paths: dict[str, str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_episodes_meta(dataset_root: Path) -> list[dict[str, Any]]:
    """Return per-episode metadata rows from either v3 flat or v3 chunked layout."""
    flat = dataset_root / "meta" / "episodes.jsonl"
    if flat.exists():
        return _read_jsonl(flat)
    chunked = dataset_root / "meta" / "episodes"
    rows: list[dict[str, Any]] = []
    if chunked.is_dir():
        for chunk_dir in sorted(chunked.glob("chunk-*")):
            for episode_file in sorted(chunk_dir.glob("episode_*.jsonl")):
                rows.extend(_read_jsonl(episode_file))
    if not rows:
        raise FileNotFoundError(f"No episode metadata found under {dataset_root}/meta")
    return rows


def _episode_language_instruction(meta_row: dict[str, Any], default: str) -> str:
    """Pick a single language instruction string from a v3 episode meta row.

    Schaeffler episodes store `tasks` as a JSON-encoded string of sub-task labels.
    We join them into a natural sentence so the VLA model sees the full sequence.
    """
    raw = meta_row.get("tasks")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = parsed
        except json.JSONDecodeError:
            return raw
    if isinstance(raw, list):
        return " then ".join(str(t).replace("_", " ") for t in raw if t)
    return default


def _video_path_template(info_features: dict[str, Any], key: str) -> str | None:
    feat = info_features.get(key)
    if not isinstance(feat, dict):
        return None
    return feat.get("videos_path") or feat.get("video_path")


def _vlm_success_map(judge_path: Path | None) -> dict[int, bool | None]:
    if judge_path is None or not judge_path.exists():
        return {}
    out: dict[int, bool | None] = {}
    for row in _read_jsonl(judge_path):
        episode_id = row.get("episode_id") or row.get("episode_index")
        if isinstance(episode_id, str) and "episode_" in episode_id:
            idx = int(episode_id.rsplit("_", 1)[-1])
        elif isinstance(episode_id, int):
            idx = episode_id
        else:
            continue
        success = row.get("outcome_success")
        out[idx] = success if isinstance(success, bool) else None
    return out


def filter_dataset(
    dataset_root: Path,
    image_keys: list[str],
    vlm_judge_path: Path | None,
    require_vlm_success: bool,
) -> tuple[list[EpisodeRecord], list[dict[str, Any]]]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    data_template = info["data_path"]
    features = info["features"]

    video_templates = {key: _video_path_template(features, key) for key in image_keys}
    missing_templates = [key for key, tpl in video_templates.items() if tpl is None]
    if missing_templates:
        raise ValueError(f"info.json has no videos_path for: {missing_templates}")

    episodes_meta = _load_episodes_meta(dataset_root)
    vlm_success = _vlm_success_map(vlm_judge_path)

    eligible: list[EpisodeRecord] = []
    skipped: list[dict[str, Any]] = []

    for row in episodes_meta:
        idx = int(row["episode_index"])
        length = int(row.get("length", 0))
        data_rel = data_template.format(episode_index=idx)
        data_path = dataset_root / data_rel
        if not data_path.exists():
            skipped.append({"episode_index": idx, "reason": "missing_data_file"})
            continue

        view_paths: dict[str, str] = {}
        missing_views: list[str] = []
        for key, tpl in video_templates.items():
            assert tpl is not None
            video_rel = tpl.format(episode_index=idx)
            video_full = dataset_root / video_rel
            if video_full.exists():
                view_paths[key] = video_rel
            else:
                missing_views.append(key)
        if missing_views:
            skipped.append({"episode_index": idx, "reason": "missing_views", "views": missing_views})
            continue

        success = vlm_success.get(idx)
        if require_vlm_success and success is False:
            skipped.append({"episode_index": idx, "reason": "vlm_judged_failure"})
            continue
        if require_vlm_success and success is None:
            skipped.append({"episode_index": idx, "reason": "no_vlm_verdict"})
            continue

        eligible.append(
            EpisodeRecord(
                episode_index=idx,
                length=length,
                language_instruction=_episode_language_instruction(row, default="manipulate object"),
                success=success,
                data_path=data_rel,
                video_paths=view_paths,
            )
        )

    return eligible, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="LeRobot v3 dataset root")
    parser.add_argument(
        "--image-keys",
        nargs="+",
        required=True,
        help="Feature keys (observation.images.*) required for each episode",
    )
    parser.add_argument(
        "--vlm-judge",
        type=Path,
        default=None,
        help="Path to VLM-judge JSONL output (used to drop judged-failure episodes)",
    )
    parser.add_argument(
        "--require-vlm-success",
        action="store_true",
        help="Drop episodes the VLM judge marked False or unjudged",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output manifest JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    eligible, skipped = filter_dataset(
        dataset_root=args.dataset,
        image_keys=list(args.image_keys),
        vlm_judge_path=args.vlm_judge,
        require_vlm_success=args.require_vlm_success,
    )

    manifest = {
        "dataset_root": str(args.dataset),
        "image_keys": list(args.image_keys),
        "require_vlm_success": args.require_vlm_success,
        "vlm_judge_path": str(args.vlm_judge) if args.vlm_judge else None,
        "n_eligible": len(eligible),
        "n_skipped": len(skipped),
        "episodes": [asdict(ep) for ep in eligible],
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))

    total_frames = sum(ep.length for ep in eligible)
    _LOGGER.info("Eligible episodes: %d (%d frames)", len(eligible), total_frames)
    _LOGGER.info("Skipped episodes:  %d", len(skipped))
    _LOGGER.info("Manifest written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
