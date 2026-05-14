"""Convert a JSONL-formatted "v3-like" LeRobot dataset to standard LeRobot v2.1 Parquet.

The Schaeffler bimanual UR5e capture pipeline emits a directory layout that
declares `codebase_version: v3.0` in `meta/info.json` but stores per-frame
records as JSONL (`data/chunk-NNN/episode_*.jsonl`) and per-episode metadata
as individual JSONL files (`meta/episodes/chunk-NNN/episode_*.jsonl`). The
Isaac-GR00T data loader (and its upstream v3->v2 converter) only handles
standard LeRobot v2.1 Parquet, so this helper bridges the gap by emitting:

    meta/info.json           codebase_version=v2.1; data_path / video_path patterns
    meta/episodes.jsonl      one JSON object per episode (episode_index, tasks, length)
    meta/tasks.jsonl         passed through unchanged
    data/chunk-NNN/episode_{NNNNNN}.parquet   per-frame records as Parquet
    videos/...               left in place (video_path pattern in info.json points at them)

An optional `training_manifest.json` (produced by the dataset analysis VLM
judge) can be supplied via `--manifest` to filter to a curated subset of
episodes. When `--manifest` is provided, only episodes listed under
`episodes[*].episode_index` with `success: true` (when present) are kept,
and the source `training_manifest.json` is copied into `meta/` for traceability.

Idempotent: running it on a dataset that already declares v2.1 is a no-op
unless `--force` is passed.

Usage:

    python training/il/scripts/gr00t/jsonl_to_lerobot_v21.py \
        --source /workspace/data/schaeffler_raw \
        --dest   /workspace/data/schaeffler_bimanual \
        --manifest /workspace/data/schaeffler_raw/training_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger("jsonl_to_lerobot_v21")

DATA_PATH_V21 = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_V21 = "videos/{video_key}/episode_{episode_index:06d}.mp4"
MIN_EPISODE_FRAMES = 32


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {ep["episode_index"]: ep for ep in data.get("episodes", [])}


def _episode_record(meta_episodes_dir: Path, episode_index: int) -> dict[str, Any]:
    """Read per-episode metadata file (one JSON object on first line)."""
    chunk = episode_index // 1000
    candidate = meta_episodes_dir / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing per-episode metadata: {candidate}")
    with candidate.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty per-episode metadata file: {candidate}")


def _resolve_tasks(record: dict[str, Any], tasks_map: dict[int, str]) -> list[str]:
    raw = record.get("tasks")
    if isinstance(raw, list) and raw:
        return [str(t) for t in raw]
    task_index = record.get("task_index")
    if task_index is not None and task_index in tasks_map:
        return [tasks_map[task_index]]
    return []


def _ffprobe_frame_count(video_path: Path) -> int:
    """Return the exact video frame count via ffprobe (-count_frames)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nk=1:nw=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout.strip()
    if not out.isdigit():
        raise RuntimeError(f"ffprobe failed for {video_path}: rc={result.returncode} stderr={result.stderr.strip()}")
    return int(out)


def _video_keys(source_info: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in source_info.get("features", {}) if key.startswith("observation.images.")
    )


def _min_video_frames(
    source: Path,
    video_keys: list[str],
    video_path_pattern: str,
    episode_index: int,
) -> int | None:
    """Return the minimum frame count across all cameras for an episode, or None if any video is missing."""
    counts: list[int] = []
    for key in video_keys:
        rel = video_path_pattern.format(video_key=key, episode_index=episode_index)
        path = source / rel
        if not path.is_file():
            _LOGGER.warning("Missing video for episode %d: %s", episode_index, path)
            return None
        counts.append(_ffprobe_frame_count(path))
    return min(counts) if counts else None


def _convert_episode(
    source_jsonl: Path,
    dest_parquet: Path,
    max_length: int | None = None,
) -> int:
    rows = list(_read_jsonl(source_jsonl))
    if not rows:
        raise ValueError(f"Episode has no frames: {source_jsonl}")
    if max_length is not None and len(rows) > max_length:
        rows = rows[:max_length]
    df = pd.DataFrame(rows)
    dest_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)
    return len(rows)


def _write_info(
    source_info: dict[str, Any],
    dest_dir: Path,
    total_episodes: int,
    total_frames: int,
) -> None:
    info = dict(source_info)
    info["codebase_version"] = "v2.1"
    info["data_path"] = DATA_PATH_V21
    info["video_path"] = VIDEO_PATH_V21
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info.setdefault("chunks_size", 1000)
    (dest_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")


def _write_episodes(dest_dir: Path, episodes: list[dict[str, Any]]) -> None:
    path = dest_dir / "meta" / "episodes.jsonl"
    with path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def _link_or_copy_videos(source: Path, dest: Path) -> None:
    src_videos = source / "videos"
    if not src_videos.is_dir():
        _LOGGER.warning("Source has no videos/ directory at %s", src_videos)
        return
    dest_videos = dest / "videos"
    if dest_videos.exists():
        return
    try:
        dest_videos.symlink_to(src_videos.resolve())
        _LOGGER.info("Symlinked videos/ -> %s", src_videos.resolve())
    except OSError:
        _LOGGER.info("Symlink failed; copying videos/ (this may take a while)")
        shutil.copytree(src_videos, dest_videos)


def convert(
    source: Path,
    dest: Path,
    manifest_path: Path | None,
    force: bool,
) -> int:
    if not source.is_dir():
        _LOGGER.error("Source not found: %s", source)
        return 2

    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        _LOGGER.error("Source missing meta/info.json: %s", info_path)
        return 2

    source_info = json.loads(info_path.read_text())

    dest_info_path = dest / "meta" / "info.json"
    if dest_info_path.exists() and not force:
        existing = json.loads(dest_info_path.read_text())
        if existing.get("codebase_version", "").startswith("v2"):
            _LOGGER.info("Dest already at %s; skipping (pass --force).", existing["codebase_version"])
            return 0

    if dest.exists() and force:
        _LOGGER.info("--force enabled: clearing %s", dest)
        shutil.rmtree(dest)

    (dest / "meta").mkdir(parents=True, exist_ok=True)

    tasks_src = source / "meta" / "tasks.jsonl"
    if not tasks_src.is_file():
        _LOGGER.error("Source missing meta/tasks.jsonl: %s", tasks_src)
        return 2
    shutil.copy2(tasks_src, dest / "meta" / "tasks.jsonl")
    tasks_map = {row["task_index"]: row["task"] for row in _read_jsonl(tasks_src)}

    meta_episodes_dir = source / "meta" / "episodes"
    if not meta_episodes_dir.is_dir():
        _LOGGER.error("Source missing meta/episodes/: %s", meta_episodes_dir)
        return 2

    manifest_index: dict[int, dict[str, Any]] | None = None
    if manifest_path is not None:
        if not manifest_path.is_file():
            _LOGGER.error("Manifest not found: %s", manifest_path)
            return 2
        manifest_index = _load_manifest(manifest_path)
        shutil.copy2(manifest_path, dest / "meta" / "training_manifest.json")
        _LOGGER.info("Loaded manifest with %d entries", len(manifest_index))

    chunks_size = int(source_info.get("chunks_size", 1000))
    data_path_pattern = source_info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl"
    )
    src_video_path_pattern = source_info.get("video_path", VIDEO_PATH_V21)
    video_keys = _video_keys(source_info)
    if not video_keys:
        _LOGGER.warning("Source info.json declares no observation.images.* features; skipping video alignment.")

    # Enumerate candidate episodes: prefer manifest order, otherwise per-episode metadata files.
    if manifest_index is not None:
        candidates = sorted(manifest_index.keys())
    else:
        candidates = []
        for chunk_dir in sorted(meta_episodes_dir.glob("chunk-*")):
            for episode_file in sorted(chunk_dir.glob("episode_*.jsonl")):
                idx = int(episode_file.stem.replace("episode_", ""))
                candidates.append(idx)

    episodes_out: list[dict[str, Any]] = []
    total_frames = 0
    for episode_index in candidates:
        manifest_entry = manifest_index.get(episode_index) if manifest_index else None
        if manifest_entry is not None and manifest_entry.get("success") is False:
            _LOGGER.debug("Skip episode %d (manifest success=false)", episode_index)
            continue
        try:
            record = _episode_record(meta_episodes_dir, episode_index)
        except FileNotFoundError as exc:
            _LOGGER.warning("Skip episode %d: %s", episode_index, exc)
            continue

        chunk_idx = episode_index // chunks_size
        source_data = source / data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        if not source_data.is_file():
            _LOGGER.warning("Skip episode %d: source data missing %s", episode_index, source_data)
            continue

        dest_data = dest / DATA_PATH_V21.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        max_length: int | None = None
        if video_keys:
            min_frames = _min_video_frames(
                source, video_keys, src_video_path_pattern, episode_index
            )
            if min_frames is None:
                _LOGGER.warning("Skip episode %d: missing video for at least one camera", episode_index)
                continue
            max_length = min_frames
            if min_frames < MIN_EPISODE_FRAMES:
                _LOGGER.warning(
                    "Skip episode %d: shortest video only %d frames (< %d minimum)",
                    episode_index, min_frames, MIN_EPISODE_FRAMES,
                )
                continue
        length = _convert_episode(source_data, dest_data, max_length=max_length)
        if max_length is not None and length < max_length:
            _LOGGER.info(
                "Episode %d: parquet shorter (%d) than min video frames (%d); using parquet length",
                episode_index, length, max_length,
            )
        tasks_list = _resolve_tasks(record, tasks_map)
        if manifest_entry and manifest_entry.get("language_instruction"):
            tasks_list = [manifest_entry["language_instruction"]]
        episodes_out.append(
            {
                "episode_index": episode_index,
                "tasks": tasks_list,
                "length": length,
            }
        )
        total_frames += length
        _LOGGER.info("Episode %d -> %s (%d frames)", episode_index, dest_data.name, length)

    if not episodes_out:
        _LOGGER.error("No episodes converted; aborting.")
        return 1

    _write_episodes(dest, episodes_out)
    _write_info(source_info, dest, total_episodes=len(episodes_out), total_frames=total_frames)
    _link_or_copy_videos(source, dest)

    _LOGGER.info(
        "Converted %d episodes (%d frames total) to %s", len(episodes_out), total_frames, dest
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source dataset directory (JSONL layout).")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory for v2.1 layout.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional training_manifest.json filter.")
    parser.add_argument("--force", action="store_true", help="Clear dest and re-convert.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[jsonl_to_lerobot_v21] %(message)s",
    )
    return convert(args.source, args.dest, args.manifest, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
