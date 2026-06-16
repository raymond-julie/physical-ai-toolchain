#!/usr/bin/env python
"""Convert MCAP teleop recordings to a LeRobot v2.1 dataset (videos, not images).

Source layout: ``<src>/<timestamp>/<timestamp>.mcap`` (one episode per directory).
Output layout: a LeRobot v2.1 dataset with:

  - observation.state  : [14] follower joint positions (left 7 + right 7)
  - action             : [14] follower MoveToPosition targets (left 7 + right 7)
  - observation.tcp_wrench.{left,right} : [6] force/torque per follower arm
  - 4 camera views encoded as MP4 video (left image of each Orbbec camera)

Each topic is sampled onto a uniform timeline at the dataset fps using the MCAP
``log_time`` (a single consistent clock) with nearest-neighbor selection.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "bimanual_ur"
DEFAULT_FPS = 30
VIDEO_CODEC = "libx264"          # encoder name for PyAV
VIDEO_CODEC_INFO_NAME = "h264"   # canonical name stored in info.json
PIX_FMT = "yuv420p"

# State: follower observed joint positions (6 joints + gripper = 7 per arm).
STATE_TOPICS = [
    ("/arm_left_follower/arm/joint_states", "position"),
    ("/arm_right_follower/arm/joint_states", "position"),
]
# Action: commanded follower target positions (7 per arm).
ACTION_TOPICS = [
    ("/arm_left_follower/action/MoveToPosition", "position"),
    ("/arm_right_follower/action/MoveToPosition", "position"),
]
# Force/torque: TCP wrench of each follower arm (force xyz + torque xyz).
WRENCH_TOPICS = {
    "observation.tcp_wrench.left": "/arm_left_follower/arm/tcp_wrench",
    "observation.tcp_wrench.right": "/arm_right_follower/arm/tcp_wrench",
}
WRENCH_NAMES = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
WRENCH_DIM = len(WRENCH_NAMES)
# Cameras: use the left image of each Orbbec camera.
CAMERAS = {
    "observation.images.cam_high": "/cam_high/camera/left_image/distorted/compressed",
    "observation.images.cam_low": "/cam_low/camera/left_image/distorted/compressed",
    "observation.images.cam_left_wrist": "/cam_left_wrist/camera/left_image/distorted/compressed",
    "observation.images.cam_right_wrist": "/cam_right_wrist/camera/left_image/distorted/compressed",
}

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
STATE_ACTION_NAMES = [f"left_{n}" for n in JOINT_NAMES] + [f"right_{n}" for n in JOINT_NAMES]
STATE_DIM = len(STATE_ACTION_NAMES)


class ConversionError(RuntimeError):
    """Raised when an MCAP recording cannot be converted to a LeRobot dataset."""


@dataclass
class TimeSeries:
    """Timestamped float vectors with nearest-neighbor resampling."""

    times: list = field(default_factory=list)   # nanosecond log_time
    values: list = field(default_factory=list)  # np.ndarray per sample

    def finalize(self) -> None:
        """Sort samples by time and convert to dense arrays."""
        order = np.argsort(self.times)
        self.times = np.asarray(self.times, dtype=np.int64)[order]
        self.values = np.asarray(self.values, dtype=np.float32)[order]

    def sample_nearest(self, grid_ns: np.ndarray) -> np.ndarray:
        """Return the nearest sample value for each timestamp in ``grid_ns``."""
        idx = np.searchsorted(self.times, grid_ns)
        idx = np.clip(idx, 1, len(self.times) - 1)
        left = self.times[idx - 1]
        right = self.times[idx]
        choose_left = (grid_ns - left) <= (right - grid_ns)
        idx = idx - choose_left.astype(int)
        idx = np.clip(idx, 0, len(self.times) - 1)
        return self.values[idx]


@dataclass
class ImageSeries:
    """Timestamped JPEG buffers with nearest-neighbor frame selection."""

    times: list = field(default_factory=list)
    jpegs: list = field(default_factory=list)

    def order(self) -> None:
        """Sort frames by time and convert timestamps to a dense array."""
        order = np.argsort(self.times)
        self.times = np.asarray(self.times, dtype=np.int64)[order]
        self.jpegs = [self.jpegs[i] for i in order]

    def nearest_indices(self, grid_ns: np.ndarray) -> np.ndarray:
        """Return the nearest frame index for each timestamp in ``grid_ns``."""
        idx = np.searchsorted(self.times, grid_ns)
        idx = np.clip(idx, 1, len(self.times) - 1)
        left = self.times[idx - 1]
        right = self.times[idx]
        choose_left = (grid_ns - left) <= (right - grid_ns)
        idx = idx - choose_left.astype(int)
        return np.clip(idx, 0, len(self.times) - 1)


def read_episode(
    mcap_path: Path,
) -> tuple[dict[str, TimeSeries], dict[str, TimeSeries], dict[str, TimeSeries], dict[str, ImageSeries]]:
    """Read state, action, wrench, and camera topics from one MCAP file."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory as Ros2Decoder

    ros2 = Ros2Decoder()
    state_series = {topic: TimeSeries() for topic, _ in STATE_TOPICS}
    action_series = {topic: TimeSeries() for topic, _ in ACTION_TOPICS}
    wrench_series = {topic: TimeSeries() for topic in WRENCH_TOPICS.values()}
    image_series = {cam: ImageSeries() for cam in CAMERAS}
    cam_topic_to_key = {topic: cam for cam, topic in CAMERAS.items()}
    action_topics = {topic for topic, _ in ACTION_TOPICS}
    wrench_topics = set(WRENCH_TOPICS.values())

    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic in state_series:
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                state_series[topic].times.append(message.log_time)
                state_series[topic].values.append(np.asarray(decoded.position, dtype=np.float32))
            elif topic in action_topics:
                obj = json.loads(message.data)
                action_series[topic].times.append(message.log_time)
                action_series[topic].values.append(np.asarray(obj["position"], dtype=np.float32))
            elif topic in wrench_topics:
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                force, torque = decoded.wrench.force, decoded.wrench.torque
                wrench_series[topic].times.append(message.log_time)
                wrench_series[topic].values.append(
                    np.asarray([force.x, force.y, force.z, torque.x, torque.y, torque.z], dtype=np.float32)
                )
            elif topic in cam_topic_to_key:
                key = cam_topic_to_key[topic]
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                image_series[key].times.append(message.log_time)
                image_series[key].jpegs.append(bytes(decoded.data))

    for series in state_series.values():
        series.finalize()
    for series in action_series.values():
        series.finalize()
    for series in wrench_series.values():
        series.finalize()
    for series in image_series.values():
        series.order()
    return state_series, action_series, wrench_series, image_series


def build_grid(
    state_series: dict[str, TimeSeries],
    action_series: dict[str, TimeSeries],
    wrench_series: dict[str, TimeSeries],
    image_series: dict[str, ImageSeries],
    fps: int,
) -> tuple[np.ndarray, int]:
    """Build a uniform timeline (ns) covering the overlap of all streams."""
    starts = []
    ends = []
    for collection in (state_series.values(), action_series.values(), wrench_series.values()):
        for series in collection:
            starts.append(int(series.times[0]))
            ends.append(int(series.times[-1]))
    for series in image_series.values():
        starts.append(int(series.times[0]))
        ends.append(int(series.times[-1]))
    t0 = max(starts)   # overlap window so every stream has data
    t1 = min(ends)
    duration_s = (t1 - t0) / 1e9
    n = max(1, int(round(duration_s * fps)))
    grid_ns = t0 + (np.arange(n) * (1e9 / fps)).astype(np.int64)
    return grid_ns, t0


def assemble_state(
    series_map: dict[str, TimeSeries],
    topics: list[tuple[str, str]],
    grid_ns: np.ndarray,
) -> np.ndarray:
    """Concatenate nearest-neighbor samples for each topic into one array."""
    cols = [series_map[topic].sample_nearest(grid_ns) for topic, _ in topics]
    return np.concatenate(cols, axis=1).astype(np.float32)


def decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode a JPEG buffer to an RGB uint8 array."""
    from PIL import Image

    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img)


def encode_video(frames_rgb: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Encode RGB frames to an H.264 MP4 at the dataset fps."""
    import av

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream(VIDEO_CODEC, rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = PIX_FMT
    stream.options = {"crf": "23", "preset": "medium"}
    stream.time_base = Fraction(1, fps)
    for arr in frames_rgb:
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def feature_stats(arr: np.ndarray) -> dict[str, list]:
    """min/max/mean/std/count for an ``[N, D]`` float array."""
    return {
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "count": [int(arr.shape[0])],
    }


def image_stats(sample_frames: np.ndarray) -> dict[str, list]:
    """Per-channel stats with shape ``(3, 1, 1)``, normalized to ``[0, 1]``."""
    x = sample_frames.astype(np.float32) / 255.0  # [N,H,W,3]
    axes = (0, 1, 2)

    def reshape(values: np.ndarray) -> list:
        return values.reshape(3, 1, 1).tolist()

    return {
        "min": reshape(x.min(axis=axes)),
        "max": reshape(x.max(axis=axes)),
        "mean": reshape(x.mean(axis=axes)),
        "std": reshape(x.std(axis=axes)),
        "count": [int(sample_frames.shape[0])],
    }


def build_features(img_h: int, img_w: int, fps: int) -> dict[str, dict]:
    """Build the LeRobot ``info.json`` feature schema for this embodiment."""
    features = {
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": {"motors": STATE_ACTION_NAMES}},
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": {"motors": STATE_ACTION_NAMES}},
    }
    for wrench_key in WRENCH_TOPICS:
        features[wrench_key] = {
            "dtype": "float32",
            "shape": [WRENCH_DIM],
            "names": {"axes": WRENCH_NAMES},
        }
    for cam in CAMERAS:
        features[cam] = {
            "dtype": "video",
            "shape": [img_h, img_w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": img_h,
                "video.width": img_w,
                "video.codec": VIDEO_CODEC_INFO_NAME,
                "video.pix_fmt": PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def convert(  # noqa: C901 — single batch pipeline kept linear for readability
    src: Path,
    out: Path,
    repo_id: str,
    task: str,
    fps: int,
    date_prefix: str | None,
    limit: int | None,
    overwrite: bool,
) -> None:
    """Convert every MCAP episode under ``src`` into a LeRobot v2.1 dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise ConversionError(f"Output {out} exists. Use --overwrite.")

    ep_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if date_prefix:
        ep_dirs = [d for d in ep_dirs if d.name.startswith(date_prefix)]
    if limit:
        ep_dirs = ep_dirs[:limit]
    if not ep_dirs:
        raise ConversionError("No episode directories found.")

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    episodes_stats = []
    total_frames = 0
    img_h = img_w = None
    ep_idx = 0
    skipped = []

    for ep_dir in tqdm(ep_dirs, desc="episodes"):
        mcap_path = ep_dir / f"{ep_dir.name}.mcap"
        if not mcap_path.exists():
            candidates = list(ep_dir.glob("*.mcap"))
            if not candidates:
                skipped.append((ep_dir.name, "no mcap"))
                continue
            mcap_path = candidates[0]
        if mcap_path.stat().st_size == 0:
            skipped.append((ep_dir.name, "empty mcap"))
            continue

        try:
            state_series, action_series, wrench_series, image_series = read_episode(mcap_path)
            grid_ns, _ = build_grid(state_series, action_series, wrench_series, image_series, fps)
        except Exception as exc:  # corrupt/incomplete recording — skip and record
            skipped.append((ep_dir.name, f"{type(exc).__name__}: {exc}"))
            continue
        n = len(grid_ns)

        state = assemble_state(state_series, STATE_TOPICS, grid_ns)     # [n,14]
        action = assemble_state(action_series, ACTION_TOPICS, grid_ns)  # [n,14]
        wrench = {
            key: wrench_series[topic].sample_nearest(grid_ns)           # [n,6]
            for key, topic in WRENCH_TOPICS.items()
        }

        ep_stats = {
            "observation.state": feature_stats(state),
            "action": feature_stats(action),
        }
        for key, arr in wrench.items():
            ep_stats[key] = feature_stats(arr)
        for cam in CAMERAS:
            iseries = image_series[cam]
            sel = iseries.nearest_indices(grid_ns)
            frames = [decode_jpeg(iseries.jpegs[i]) for i in sel]
            if img_h is None:
                img_h, img_w = frames[0].shape[:2]
            vid_path = out / "videos" / "chunk-000" / cam / f"episode_{ep_idx:06d}.mp4"
            encode_video(frames, vid_path, fps)
            sidx = np.linspace(0, n - 1, min(n, 30)).astype(int)
            ep_stats[cam] = image_stats(np.stack([frames[i] for i in sidx]))

        timestamp = (grid_ns - grid_ns[0]).astype(np.float64) / 1e9
        columns = {
            "action": pa.array(list(action), type=pa.list_(pa.float32(), STATE_DIM)),
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), STATE_DIM)),
        }
        for key, arr in wrench.items():
            columns[key] = pa.array(list(arr), type=pa.list_(pa.float32(), WRENCH_DIM))
        columns.update(
            {
                "timestamp": pa.array(timestamp.astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64)),
                "index": pa.array(np.arange(total_frames, total_frames + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            }
        )
        table = pa.table(columns)
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n})
        episodes_stats.append({"episode_index": ep_idx, "stats": ep_stats})
        total_frames += n
        ep_idx += 1

    n_eps = len(episodes_meta)

    with (out / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w") as f:
        for episode in episodes_meta:
            f.write(json.dumps(episode) + "\n")
    with (out / "meta" / "episodes_stats.jsonl").open("w") as f:
        for episode in episodes_stats:
            f.write(json.dumps(episode) + "\n")

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "repo_id": repo_id,
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_eps * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features(img_h, img_w, fps),
    }
    with (out / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=4)

    _LOGGER.info("Done: %d episodes, %d frames -> %s", n_eps, total_frames, out)
    if skipped:
        _LOGGER.warning("Skipped %d episode(s):", len(skipped))
        for name, reason in skipped:
            _LOGGER.warning("  - %s: %s", name, reason)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("/home/aloha/recordings"))
    parser.add_argument("--out", type=Path, required=True, help="output dataset root")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--date-prefix",
        default=None,
        help="only convert episode dirs starting with this prefix, e.g. 20260603",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        convert(
            args.src,
            args.out,
            args.repo_id,
            args.task,
            args.fps,
            args.date_prefix,
            args.limit,
            args.overwrite,
        )
    except ConversionError:
        _LOGGER.exception("Conversion failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
