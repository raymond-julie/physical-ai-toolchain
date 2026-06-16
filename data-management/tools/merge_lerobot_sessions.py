#!/usr/bin/env python3
"""Merge multiple LeRobot v2.1 session folders into one combined dataset.

Reads every ``session_*`` folder under the source directory, re-indexes episodes
and frames globally, copies videos, and writes a single unified LeRobot v2.1
dataset to the output path.

Usage:
    python merge_lerobot_sessions.py ./downloads ./combined-sessions
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class MergeError(RuntimeError):
    """Raised when sessions cannot be merged into a single dataset."""


def discover_sessions(src_dir: Path) -> list[Path]:
    """Return sorted ``session_*`` subdirectories of ``src_dir``."""
    sessions = sorted(p for p in src_dir.iterdir() if p.is_dir() and p.name.startswith("session_"))
    if not sessions:
        raise MergeError(f"No session_* folders found in {src_dir}")
    return sessions


def discover_video_keys(info: dict) -> list[str]:
    """Return the feature keys whose dtype is ``video`` from a dataset info dict."""
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def reindex_episode_entry(entry: dict, new_index: int) -> dict:
    """Return a copy of an episode metadata entry with a new ``episode_index``."""
    updated = copy.deepcopy(entry)
    updated["episode_index"] = new_index
    return updated


def frame_index_range(start_index: int, n_rows: int) -> list[int]:
    """Return the contiguous global frame indices for one episode."""
    return list(range(start_index, start_index + n_rows))


def build_combined_info(ref_info: dict, total_episodes: int, total_frames: int, total_videos: int) -> dict:
    """Return reference info updated with merged episode/frame/video totals."""
    combined = copy.deepcopy(ref_info)
    combined["total_episodes"] = total_episodes
    combined["total_frames"] = total_frames
    combined["total_videos"] = total_videos
    combined["splits"] = {"train": f"0:{total_episodes}"}
    return combined


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of dicts (empty when missing)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]


def merge_sessions(src_dir: Path, out_dir: Path) -> None:
    """Merge every ``session_*`` folder under ``src_dir`` into ``out_dir``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sessions = discover_sessions(src_dir)
    _LOGGER.info("Found %d sessions to merge", len(sessions))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    ref_info = json.loads((sessions[0] / "meta" / "info.json").read_text())
    ref_modality = json.loads((sessions[0] / "meta" / "modality.json").read_text())
    ref_tasks = (sessions[0] / "meta" / "tasks.jsonl").read_text().strip()

    video_keys = discover_video_keys(ref_info)
    _LOGGER.info("Video keys: %s", video_keys)
    for video_key in video_keys:
        (out_dir / "videos" / "chunk-000" / video_key).mkdir(parents=True, exist_ok=True)

    global_episode_idx = 0
    global_frame_idx = 0
    all_episodes_jsonl = []
    all_episodes_stats = []
    total_frames = 0
    total_videos = 0

    for session in sessions:
        session_info = json.loads((session / "meta" / "info.json").read_text())
        n_episodes = session_info["total_episodes"]
        _LOGGER.info("  %s: %d episodes, %d frames", session.name, n_episodes, session_info["total_frames"])

        session_episodes = _read_jsonl(session / "meta" / "episodes.jsonl")
        session_stats = _read_jsonl(session / "meta" / "episodes_stats.jsonl")

        for local_ep_idx in range(n_episodes):
            src_parquet = session / "data" / "chunk-000" / f"episode_{local_ep_idx:06d}.parquet"
            if src_parquet.exists():
                table = pq.read_table(src_parquet)
                n_rows = table.num_rows
                new_ep_col = pa.array([global_episode_idx] * n_rows, type=pa.int64())
                new_idx_col = pa.array(frame_index_range(global_frame_idx, n_rows), type=pa.int64())

                col_names = table.column_names
                if "episode_index" in col_names:
                    table = table.set_column(col_names.index("episode_index"), "episode_index", new_ep_col)
                if "index" in col_names:
                    table = table.set_column(col_names.index("index"), "index", new_idx_col)

                dst_parquet = out_dir / "data" / "chunk-000" / f"episode_{global_episode_idx:06d}.parquet"
                pq.write_table(table, dst_parquet)
                total_frames += n_rows
                global_frame_idx += n_rows

            for video_key in video_keys:
                src_video = session / "videos" / "chunk-000" / video_key / f"episode_{local_ep_idx:06d}.mp4"
                if src_video.exists():
                    dst_video = out_dir / "videos" / "chunk-000" / video_key / f"episode_{global_episode_idx:06d}.mp4"
                    shutil.copy2(src_video, dst_video)
                    total_videos += 1

            if local_ep_idx < len(session_episodes):
                entry = reindex_episode_entry(session_episodes[local_ep_idx], global_episode_idx)
                all_episodes_jsonl.append(json.dumps(entry))

            if local_ep_idx < len(session_stats):
                stat_entry = reindex_episode_entry(session_stats[local_ep_idx], global_episode_idx)
                all_episodes_stats.append(json.dumps(stat_entry))

            global_episode_idx += 1

    combined_info = build_combined_info(ref_info, global_episode_idx, total_frames, total_videos)
    (out_dir / "meta" / "info.json").write_text(json.dumps(combined_info, indent=2))
    (out_dir / "meta" / "modality.json").write_text(json.dumps(ref_modality, indent=2))
    (out_dir / "meta" / "tasks.jsonl").write_text(ref_tasks + "\n")
    (out_dir / "meta" / "episodes.jsonl").write_text("\n".join(all_episodes_jsonl) + "\n")
    if all_episodes_stats:
        (out_dir / "meta" / "episodes_stats.jsonl").write_text("\n".join(all_episodes_stats) + "\n")

    _LOGGER.info(
        "Done. Combined dataset: %d episodes, %d frames, %d videos -> %s",
        global_episode_idx,
        total_frames,
        total_videos,
        out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Directory containing session_* folders")
    parser.add_argument("output_dir", type=Path, help="Output dataset path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        merge_sessions(args.source_dir, args.output_dir)
    except MergeError:
        _LOGGER.exception("Merge failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
