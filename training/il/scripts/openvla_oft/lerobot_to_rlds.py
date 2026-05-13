"""LeRobot v3 -> RLDS (TFDS) converter for OpenVLA-OFT fine-tuning.

This module exports `SchaefflerBimanualBuilder`, a `tfds.core.GeneratorBasedBuilder`
that reads the manifest emitted by `filter_dataset.py` and yields RLDS-format
episodes consumable by `vla-scripts/finetune.py` in the OpenVLA-OFT repo.

The output dataset uses the **3-image + proprio** schema expected by OFT's
ALOHA recipe:
- `observation.image`            -> primary camera (configurable)
- `observation.wrist_image_left` -> left wrist camera
- `observation.wrist_image_right`-> right wrist camera
- `observation.state`            -> 12-DOF proprio (R/L joint 1..6)
- `action`                       -> 12-DOF absolute joint angle action
- `language_instruction`         -> per-episode language string

Images are decoded with `decord`, resized to 256x256, and JPEG-encoded inside the
TFDS shard. State and action are stored as float32. Episodes longer than the
configured `max_episode_length` are truncated from the start.

The conversion is invoked either via TFDS CLI (`tfds build`) after dropping this
file under a directory layout matching `rlds_dataset_builder`, or directly by
running this module as a script in `--dry-run` mode to verify the manifest is
loadable.

Usage (dry-run, no TFDS write):
    python -m training.il.scripts.openvla_oft.lerobot_to_rlds \
        --manifest datasets/schaeffler_sim_avc1/second_collection/training_manifest.json \
        --primary-camera observation.images.d405_stationary_r_0 \
        --left-wrist observation.images.d405_stationary_l_1 \
        --right-wrist observation.images.d405_stationary_l_2 \
        --dry-run

Usage (full RLDS build):
    python -m training.il.scripts.openvla_oft.lerobot_to_rlds \
        --manifest datasets/schaeffler_sim_avc1/second_collection/training_manifest.json \
        --primary-camera observation.images.d405_stationary_r_0 \
        --left-wrist observation.images.d405_stationary_l_1 \
        --right-wrist observation.images.d405_stationary_l_2 \
        --output-dir datasets/schaeffler_sim_avc1/rlds
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_LOGGER = logging.getLogger(__name__)

# ----- Constants matching OFT ALOHA preprocessing -----

IMAGE_SIZE = 256  # OFT downsizes ALOHA frames from 480x640 to 256x256
ACTION_DIM = 12  # Schaeffler dual-arm UR5e (no gripper)
STATE_DIM = 12


@dataclass(frozen=True)
class CameraMapping:
    primary: str
    left_wrist: str
    right_wrist: str


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text())
    if "episodes" not in data:
        raise ValueError(f"Manifest at {manifest_path} is missing 'episodes'")
    return data


def _decode_video(path: Path, target_size: int) -> Any:
    """Decode all frames from `path` and return a numpy array (T, H, W, 3) uint8."""
    import decord  # heavy optional dep, imported lazily
    import numpy as np

    reader = decord.VideoReader(str(path), width=target_size, height=target_size)
    frames = reader.get_batch(range(len(reader))).asnumpy()
    return np.ascontiguousarray(frames, dtype=np.uint8)


def _iter_episode_steps(
    dataset_root: Path,
    episode: dict[str, Any],
    cameras: CameraMapping,
) -> Iterator[dict[str, Any]]:
    """Yield one RLDS step dict per data row in the episode."""
    import numpy as np

    data_rows = [
        json.loads(line) for line in (dataset_root / episode["data_path"]).read_text().splitlines() if line.strip()
    ]

    primary = _decode_video(dataset_root / episode["video_paths"][cameras.primary], IMAGE_SIZE)
    left = _decode_video(dataset_root / episode["video_paths"][cameras.left_wrist], IMAGE_SIZE)
    right = _decode_video(dataset_root / episode["video_paths"][cameras.right_wrist], IMAGE_SIZE)

    n = min(len(data_rows), len(primary), len(left), len(right))
    if n != len(data_rows):
        _LOGGER.warning(
            "Episode %d data/video length mismatch: data=%d primary=%d left=%d right=%d (truncating to %d)",
            episode["episode_index"],
            len(data_rows),
            len(primary),
            len(left),
            len(right),
            n,
        )

    language = episode["language_instruction"]

    for i in range(n):
        row = data_rows[i]
        state = np.asarray(row["observation.state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)
        yield {
            "observation": {
                "image": primary[i],
                "wrist_image_left": left[i],
                "wrist_image_right": right[i],
                "state": state,
            },
            "action": action,
            "discount": 1.0,
            "reward": float(i == n - 1),
            "is_first": i == 0,
            "is_last": i == n - 1,
            "is_terminal": i == n - 1,
            "language_instruction": language,
        }


def make_builder_class(
    name: str,
    dataset_root: Path,
    manifest_path: Path,
    cameras: CameraMapping,
    val_fraction: float = 0.05,
) -> type:
    """Return a dynamically-defined `tfds.core.GeneratorBasedBuilder` subclass.

    Lazy-imports `tensorflow_datasets` so the rest of this module is usable
    in a `--dry-run` workflow without TF installed.
    """
    import tensorflow_datasets as tfds

    manifest = _load_manifest(manifest_path)

    class SchaefflerBimanualBuilder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Initial release from LeRobot v3 manifest."}

        def _info(self) -> tfds.core.DatasetInfo:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                            dtype=tfds.features.Image.np_dtype,
                                            encoding_format="jpeg",
                                        ),
                                        "wrist_image_left": tfds.features.Image(
                                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                            dtype=tfds.features.Image.np_dtype,
                                            encoding_format="jpeg",
                                        ),
                                        "wrist_image_right": tfds.features.Image(
                                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                            dtype=tfds.features.Image.np_dtype,
                                            encoding_format="jpeg",
                                        ),
                                        "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype="float32"),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype="float32"),
                                "discount": tfds.features.Scalar(dtype="float32"),
                                "reward": tfds.features.Scalar(dtype="float32"),
                                "is_first": tfds.features.Scalar(dtype="bool"),
                                "is_last": tfds.features.Scalar(dtype="bool"),
                                "is_terminal": tfds.features.Scalar(dtype="bool"),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "episode_index": tfds.features.Scalar(dtype="int64"),
                                "length": tfds.features.Scalar(dtype="int64"),
                                "success": tfds.features.Scalar(dtype="bool"),
                            }
                        ),
                    }
                ),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager):  # noqa: ARG002 (TFDS API)
            episodes = manifest["episodes"]
            split_idx = max(1, int(len(episodes) * (1.0 - val_fraction)))
            return {
                "train": self._generate_examples(episodes[:split_idx]),
                "val": self._generate_examples(episodes[split_idx:]),
            }

        def _generate_examples(self, episodes: list[dict[str, Any]]):
            for episode in episodes:
                steps = list(_iter_episode_steps(dataset_root, episode, cameras))
                yield episode["episode_index"], {
                    "steps": steps,
                    "episode_metadata": {
                        "episode_index": int(episode["episode_index"]),
                        "length": int(episode["length"]),
                        "success": bool(episode.get("success", True)),
                    },
                }

    SchaefflerBimanualBuilder.__name__ = name
    return SchaefflerBimanualBuilder


def _dry_run(manifest_path: Path, cameras: CameraMapping, dataset_root: Path) -> int:
    manifest = _load_manifest(manifest_path)
    episodes = manifest["episodes"]
    _LOGGER.info("Manifest reports %d eligible episodes", len(episodes))

    sample = episodes[0]
    _LOGGER.info(
        "Sample episode %d: %d frames, instruction=%r",
        sample["episode_index"],
        sample["length"],
        sample["language_instruction"],
    )
    for label, key in (("primary", cameras.primary), ("left_wrist", cameras.left_wrist), ("right_wrist", cameras.right_wrist)):
        rel = sample["video_paths"].get(key)
        if rel is None:
            _LOGGER.error("Sample episode is missing video for %s (%s)", label, key)
            return 1
        path = dataset_root / rel
        if not path.exists():
            _LOGGER.error("Sample episode video missing on disk: %s", path)
            return 1
        _LOGGER.info("  %s = %s (%.1f MB)", label, path, path.stat().st_size / 1e6)

    data_path = dataset_root / sample["data_path"]
    if not data_path.exists():
        _LOGGER.error("Sample data file missing: %s", data_path)
        return 1
    n_data = sum(1 for line in data_path.read_text().splitlines() if line.strip())
    _LOGGER.info("  data file rows: %d", n_data)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest from filter_dataset.py")
    parser.add_argument("--primary-camera", required=True, help="Feature key for the primary (third-person-ish) camera")
    parser.add_argument("--left-wrist", required=True, help="Feature key for the left wrist camera")
    parser.add_argument("--right-wrist", required=True, help="Feature key for the right wrist camera")
    parser.add_argument("--name", default="schaeffler_bimanual", help="TFDS dataset name")
    parser.add_argument("--output-dir", type=Path, default=None, help="TFDS data_dir for `tfds build` output")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest + paths without building TFDS")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    manifest = _load_manifest(args.manifest)
    dataset_root = Path(manifest["dataset_root"])
    cameras = CameraMapping(args.primary_camera, args.left_wrist, args.right_wrist)

    if args.dry_run:
        return _dry_run(args.manifest, cameras, dataset_root)

    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        _LOGGER.error("tensorflow_datasets is required for full build (--dry-run skips it). %s", exc)
        return 2

    builder_cls = make_builder_class(args.name, dataset_root, args.manifest, cameras, val_fraction=args.val_fraction)
    builder = builder_cls(data_dir=str(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(register_checksums=False, max_examples_per_split=None)
    )
    _LOGGER.info("RLDS dataset written under %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
