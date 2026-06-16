#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset to the GR00T-flavored LeRobot v2.0 layout.

v3.0 packs many episodes per parquet/mp4 file
(``data/chunk-{C:03d}/file-{F:03d}.parquet``,
``videos/{key}/chunk-{C:03d}/file-{F:03d}.mp4``) and keeps the episode->file
mapping plus per-episode video timestamp ranges in
``meta/episodes/chunk-{C:03d}/file-{F:03d}.parquet``.

GR00T's ``LeRobotSingleDataset`` only understands the v2.0 shape:

    data/chunk-{C:03d}/episode_{E:06d}.parquet
    videos/chunk-{C:03d}/{video_key}/episode_{E:06d}.mp4
    meta/{info,episodes,tasks,modality}.json[l]

So this tool:

  1. Splits each multi-episode data parquet into per-episode parquets using
     ``dataset_from_index``/``dataset_to_index`` from the episodes meta.
  2. For each (episode, video_key), ``ffmpeg -ss FROM -to TO`` slices the source
     chunk mp4 (re-encoded with libx264 so cuts are accurate even on non-keyframe
     boundaries -- stream-copy would otherwise insert black or repeated frames).
  3. Rewrites info.json with the v2.0 codebase_version and path templates.
  4. Synthesizes episodes.jsonl, tasks.jsonl, and modality.json from v3 metadata.

The modality.json defaults to a UR10e single-arm 7-DoF embodiment with two RGB
cameras (``observation.images.color`` and ``observation.images.color2``). Pass
``--modality-config PATH`` to supply a JSON spec for a different embodiment.

Usage:
    python convert_lerobot_v3_to_v2.py \\
        --src /data/hybrid-hack-vla-train-full \\
        --dst /data/hybrid-hack-vla-train-full-v2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# Default modality.json: UR10e single-arm, 7-DoF (6 joints + 1 gripper), two RGB cameras.
UR10E_SINGLE_ARM_MODALITY = {
    "state": {
        "single_arm": {"start": 0, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "action": {
        "single_arm": {"start": 0, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "color": {"original_key": "observation.images.color"},
        "color2": {"original_key": "observation.images.color2"},
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}


class ConversionError(RuntimeError):
    """Raised when a v3.0 dataset cannot be converted to the v2.0 layout."""


class ModalityError(ConversionError):
    """Raised when the modality spec does not match the dataset dimensions."""


@lru_cache(maxsize=1)
def _ffmpeg_exe() -> str:
    """Locate the ffmpeg binary bundled with the imageio-ffmpeg wheel."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _max_group_end(groups: dict) -> int:
    """Largest ``end`` index across a modality state/action group mapping."""
    return max(int(group["end"]) for group in groups.values())


def load_modality_spec(path: Path | None) -> dict:
    """Load a modality spec from JSON, or return the UR10e single-arm default."""
    if path is None:
        return copy.deepcopy(UR10E_SINGLE_ARM_MODALITY)
    return json.loads(path.read_text())


def build_modality_json(features: dict, spec: dict) -> dict:
    """Validate a modality spec against the dataset state/action dimensions.

    The spec is returned unchanged when its state/action groups cover exactly the
    dataset's reported dimensions; otherwise a ``ModalityError`` is raised so the
    caller can supply a matching ``--modality-config``.
    """
    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    expected_state = _max_group_end(spec["state"])
    expected_action = _max_group_end(spec["action"])
    if state_dim != expected_state or action_dim != expected_action:
        raise ModalityError(
            f"Modality spec covers state={expected_state}, action={expected_action} dims, "
            f"but the dataset reports state={state_dim}, action={action_dim}. "
            f"Pass --modality-config with a matching embodiment spec."
        )
    return spec


def build_info_json(src_info: dict, num_episodes: int, num_chunks: int) -> dict:
    """Strip v3-specific fields and add the v2.0 path templates."""
    return {
        "codebase_version": "v2.0",
        "robot_type": src_info.get("robot_type", "unknown"),
        "total_episodes": src_info.get("total_episodes", num_episodes),
        "total_frames": src_info.get("total_frames"),
        "total_tasks": src_info.get("total_tasks"),
        "total_videos": src_info.get("total_videos"),
        "total_chunks": num_chunks,
        "chunks_size": src_info.get("chunks_size", 1000),
        "fps": src_info.get("fps"),
        "splits": src_info.get("splits", {"train": f"0:{num_episodes}"}),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": src_info["features"],
    }


def _ffmpeg_cut(src: Path, dst: Path, t_from: float, t_to: float) -> None:
    """Re-encode a slice of ``src`` from ``t_from`` to ``t_to`` seconds into ``dst``.

    Re-encoding (vs ``-c copy``) keeps clip starts on a keyframe even when the
    requested cut point is not aligned with the source GOP. libx264 + crf 23 +
    veryfast is fast on a desktop CPU and visually indistinguishable from source.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-loglevel", "error",
        "-accurate_seek",
        "-i", str(src),
        "-ss", f"{t_from:.6f}",
        "-to", f"{t_to:.6f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ConversionError(f"ffmpeg failed for {dst}:\n{result.stderr}")


def convert(src_root: Path, dst_root: Path, jobs: int, modality_path: Path | None = None) -> None:
    """Convert a v3.0 dataset at ``src_root`` into a v2.0 dataset at ``dst_root``."""
    import pandas as pd
    import pyarrow.parquet as pq

    src_meta = src_root / "meta"
    info = json.loads((src_meta / "info.json").read_text())
    _LOGGER.info(
        "source: codebase=%s robot=%s episodes=%s frames=%s",
        info["codebase_version"],
        info.get("robot_type"),
        info.get("total_episodes"),
        info.get("total_frames"),
    )
    if not info["codebase_version"].startswith("v3"):
        _LOGGER.warning("Source codebase_version=%s; conversion targets v3->v2.", info["codebase_version"])

    ep_files = sorted((src_meta / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise ConversionError(f"No episode metadata under {src_meta / 'episodes'}")
    ep_df = pd.concat([pq.read_table(f).to_pandas() for f in ep_files], ignore_index=True)
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    _LOGGER.info("episodes loaded: %d", len(ep_df))

    tasks_df = pq.read_table(src_meta / "tasks.parquet").to_pandas().reset_index(drop=True)
    if "task_index" not in tasks_df.columns:
        tasks_df["task_index"] = list(range(len(tasks_df)))
    if "task" not in tasks_df.columns:
        tasks_df["task"] = tasks_df.index.astype(str)
    _LOGGER.info("tasks: %d", len(tasks_df))

    data_root_src = src_root / "data"
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for parq in sorted(data_root_src.rglob("file-*.parquet")):
        chunk = int(parq.parent.name.split("-")[-1])
        fidx = int(parq.stem.split("-")[-1])
        data_cache[(chunk, fidx)] = pq.read_table(parq).to_pandas()
    _LOGGER.info("data parquets loaded: %d", len(data_cache))

    video_keys = sorted(p.name for p in (src_root / "videos").iterdir() if p.is_dir())
    _LOGGER.info("video keys: %s", video_keys)

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "meta").mkdir(exist_ok=True)
    (dst_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for video_key in video_keys:
        (dst_root / "videos" / "chunk-000" / video_key).mkdir(parents=True, exist_ok=True)

    _LOGGER.info("splitting per-episode parquets...")
    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        chunk = int(row["data/chunk_index"])
        fidx = int(row["data/file_index"])
        lo = int(row["dataset_from_index"])
        hi = int(row["dataset_to_index"])
        df = data_cache[(chunk, fidx)]
        # The data parquet `index` column is the global frame index across the
        # shard; the episode meta from/to indices live in that same space.
        ep_rows = df[(df["index"] >= lo) & (df["index"] < hi)].copy()
        if len(ep_rows) != int(row["length"]):
            raise ConversionError(f"episode {ep_idx}: expected {row['length']} rows, got {len(ep_rows)}")
        out_path = dst_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        ep_rows.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    cut_jobs = []
    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        for video_key in video_keys:
            chunk = int(row[f"videos/{video_key}/chunk_index"])
            fidx = int(row[f"videos/{video_key}/file_index"])
            t_from = float(row[f"videos/{video_key}/from_timestamp"])
            t_to = float(row[f"videos/{video_key}/to_timestamp"])
            src_mp4 = src_root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"
            dst_mp4 = dst_root / "videos" / "chunk-000" / video_key / f"episode_{ep_idx:06d}.mp4"
            cut_jobs.append((src_mp4, dst_mp4, t_from, t_to, ep_idx, video_key))

    _LOGGER.info("re-encoding %d episode video clips with ffmpeg (jobs=%d)...", len(cut_jobs), jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_ffmpeg_cut, src_mp4, dst_mp4, t_from, t_to): (ep_idx, video_key)
            for (src_mp4, dst_mp4, t_from, t_to, ep_idx, video_key) in cut_jobs
        }
        for future in as_completed(futures):
            ep_idx, video_key = futures[future]
            future.result()
            done += 1
            if done % 20 == 0 or done == len(cut_jobs):
                _LOGGER.info("  [%d/%d] %s ep=%d", done, len(cut_jobs), video_key, ep_idx)

    _LOGGER.info("writing meta files...")
    out_meta = dst_root / "meta"

    out_info = build_info_json(info, num_episodes=len(ep_df), num_chunks=1)
    (out_meta / "info.json").write_text(json.dumps(out_info, indent=2))

    with (out_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for _, row in ep_df.iterrows():
            tasks = list(row["tasks"]) if hasattr(row["tasks"], "__iter__") else [row["tasks"]]
            f.write(
                json.dumps(
                    {
                        "episode_index": int(row["episode_index"]),
                        "tasks": tasks,
                        "length": int(row["length"]),
                    }
                )
                + "\n"
            )

    with (out_meta / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for _, trow in tasks_df.iterrows():
            f.write(json.dumps({"task_index": int(trow["task_index"]), "task": str(trow["task"])}) + "\n")

    spec = load_modality_spec(modality_path)
    modality = build_modality_json(info["features"], spec)
    (out_meta / "modality.json").write_text(json.dumps(modality, indent=2))

    # stats.json is intentionally skipped: the v3 stats arrays do not match the
    # v2 per-dim convention GR00T expects, and gr00t.data.dataset recomputes
    # valid statistics from the parquet files on first dataset init when absent.
    _LOGGER.info("done. dataset at %s", dst_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel ffmpeg workers (default 4; bump on big CPUs)")
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=None,
        help="JSON modality spec; defaults to the UR10e single-arm 7-DoF, 2-camera embodiment.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        convert(args.src.resolve(), args.dst.resolve(), args.jobs, args.modality_config)
    except ConversionError:
        _LOGGER.exception("Conversion failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
