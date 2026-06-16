#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tyro>=0.8",
# ]
# ///
"""Fine-tune openpi (pi0 / pi0.5) on the UR10e LeRobot dataset.

Mirrors the shape of ``train_gr00t_ur10e.py``. Runs from inside a checked-out
openpi repo (``/opt/openpi`` in the K8s job, see ``openpi-train.yaml``) so it can
``import openpi.*`` directly.

Flow:
    1. Sanity-check the LeRobot dataset at $DATASET_PATH.
    2. Programmatically build a TrainConfig (see ``openpi_ur10e_policy.py``) and
       register it with openpi's config registry under a unique name.
    3. Compute normalization statistics (``scripts/compute_norm_stats.py``).
    4. Launch training (``scripts/train.py``) via JAX.

Inference, post-training:
    uv run scripts/serve_policy.py policy:checkpoint \\
      --policy.config=pi05_ur10e \\
      --policy.dir=<output>/pi05_ur10e/<exp>/<step>

Usage:
    python train_openpi_ur10e.py \\
      --dataset /data/combined-sessions \\
      --output  /outputs/openpi-ur10e \\
      --exp-name run01 \\
      --max-steps 30000 --batch-size 32 --pi05 --lora
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOGGER = logging.getLogger(__name__)

# Defaults (overridable via env / CLI; align with openpi-train.yaml).
DEFAULT_DATASET = Path(os.environ.get("DATASET_PATH", "/data"))
DEFAULT_OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/outputs/openpi-ur10e"))
DEFAULT_OPENPI_DIR = Path(os.environ.get("OPENPI_DIR", "/opt/openpi"))
DEFAULT_EXP_NAME = os.environ.get("EXP_NAME", "ur10e_run")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to LeRobot v2.x dataset root (must contain meta/info.json).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory where checkpoints and assets are written.")
    parser.add_argument("--openpi-dir", type=Path, default=DEFAULT_OPENPI_DIR,
                        help="Path to a cloned openpi repo (must contain scripts/train.py).")
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME,
                        help="Run name (-> <output>/<config_name>/<exp_name>/).")
    parser.add_argument("--repo-id", default=None,
                        help="LeRobot repo id; defaults to the dataset directory path.")
    parser.add_argument("--pi05", action="store_true", default=True,
                        help="Fine-tune pi0.5 base (default; recommended).")
    parser.add_argument("--pi0", dest="pi05", action="store_false",
                        help="Fine-tune pi0 base instead of pi0.5.")
    parser.add_argument("--lora", action="store_true",
                        help="LoRA fine-tune (~22 GB VRAM); otherwise full FT (~70 GB).")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=1,
                        help="Shard model across this many devices (FSDP).")
    parser.add_argument("--default-prompt", default=None,
                        help="Override the prompt if dataset tasks.jsonl is missing.")
    parser.add_argument("--prompt-from-task", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing checkpoint dir for this exp name.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint for this exp name.")
    parser.add_argument("--skip-norm-stats", action="store_true",
                        help="Skip the compute_norm_stats step (only safe on resume).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved commands without launching training.")
    return parser


def _validate_dataset(path: Path) -> dict:
    """Verify the dataset is a LeRobot v2.x tree with the keys we expect."""
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"meta/info.json not found under {path}; not a LeRobot dataset")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    required = ["observation.state", "action", "observation.images.color"]
    missing = [k for k in required if k not in features]
    if missing:
        sys.exit(f"Dataset {path} missing required features: {missing}")
    _LOGGER.info(
        "Dataset OK: robot=%s fps=%s episodes=%s frames=%s features=%s",
        info.get("robot_type"),
        info.get("fps"),
        info.get("total_episodes"),
        info.get("total_frames"),
        sorted(features.keys()),
    )
    return info


def _register_config(args: argparse.Namespace, repo_id: str) -> Any:
    """Import openpi_ur10e_policy and register a TrainConfig with openpi."""
    # Ensure openpi (cloned at args.openpi_dir) and the policy module are importable.
    if str(args.openpi_dir / "src") not in sys.path:
        sys.path.insert(0, str(args.openpi_dir / "src"))
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import openpi_ur10e_policy as ur10e

    cfg = ur10e.build_train_configs(
        repo_id=repo_id,
        exp_name=args.exp_name,
        num_train_steps=args.max_steps,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        lora=args.lora,
        pi05=args.pi05,
        prompt_from_task=args.prompt_from_task,
        default_prompt=args.default_prompt,
    )
    ur10e.register(cfg)
    _LOGGER.info("Registered openpi TrainConfig: name=%s exp_name=%s lora=%s pi05=%s",
                 cfg.name, cfg.exp_name, args.lora, args.pi05)
    return cfg


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], *, dry: bool) -> None:
    _LOGGER.info("$ (cd %s && %s)", cwd, " ".join(cmd))
    if dry:
        return
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = create_parser().parse_args(argv)

    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.openpi_dir = args.openpi_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    if not (args.openpi_dir / "scripts" / "train.py").exists():
        sys.exit(f"openpi clone not found at {args.openpi_dir} (missing scripts/train.py)")

    _validate_dataset(args.dataset)

    # Default repo_id to the dataset directory (LeRobotDataset accepts a path).
    repo_id = args.repo_id or str(args.dataset)

    cfg = _register_config(args, repo_id=repo_id)

    # Persist asset & checkpoint dirs under args.output so artifacts survive.
    common_args = [
        f"--assets-base-dir={args.output / 'assets'}",
        f"--checkpoint-base-dir={args.output / 'checkpoints'}",
        f"--exp-name={args.exp_name}",
    ]

    env = os.environ.copy()
    # JAX needs an aggressive GPU mem fraction to fit pi0/0.5 + activations.
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    env.setdefault("HF_HOME", str(args.output / "hf-cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(args.output / "hf-cache"))
    # Make the policy module importable from openpi's worker processes too.
    extra_py = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = extra_py + os.pathsep + env.get("PYTHONPATH", "")

    # Step 1: compute normalization statistics. compute_norm_stats imports
    # openpi.training.config, so the config registration must survive across the
    # subprocess; re-register via an inline bootstrap before handing off.
    if not args.skip_norm_stats:
        _LOGGER.info("--- Computing norm stats for config: %s ---", cfg.name)
        bootstrap = (
            "import sys, runpy; "
            f"sys.path.insert(0, {extra_py!r}); "
            f"sys.path.insert(0, {str(args.openpi_dir / 'src')!r}); "
            "import openpi_ur10e_policy as u; "
            "u.register(u.build_train_configs("
            f"  repo_id={repo_id!r}, exp_name={args.exp_name!r},"
            f"  num_train_steps={args.max_steps}, batch_size={args.batch_size},"
            f"  save_interval={args.save_interval}, lora={args.lora}, pi05={args.pi05},"
            f"  prompt_from_task={args.prompt_from_task},"
            f"  default_prompt={args.default_prompt!r})); "
            f"sys.argv = ['compute_norm_stats.py', '--config-name={cfg.name}',"
            f"            '--assets-base-dir={args.output / 'assets'}']; "
            "runpy.run_path('scripts/compute_norm_stats.py', run_name='__main__')"
        )
        _run([sys.executable, "-c", bootstrap], cwd=args.openpi_dir, env=env, dry=args.dry_run)
    else:
        _LOGGER.info("Skipping compute_norm_stats (--skip-norm-stats)")

    # Step 2: train.
    _LOGGER.info("--- Launching openpi training: %s ---", cfg.name)
    train_argv_tail = [cfg.name, *common_args]
    if args.fsdp_devices and args.fsdp_devices > 1:
        train_argv_tail.append(f"--fsdp-devices={args.fsdp_devices}")
    if args.overwrite:
        train_argv_tail.append("--overwrite")
    if args.resume:
        train_argv_tail.append("--resume")
    if not args.dry_run:
        # Register the config in-process, then hand off to scripts/train.py via
        # runpy so tyro's CLI lookup finds the config.
        bootstrap = (
            "import sys, runpy; "
            f"sys.path.insert(0, {extra_py!r}); "
            f"sys.path.insert(0, {str(args.openpi_dir / 'src')!r}); "
            "import openpi_ur10e_policy as u; "
            "u.register(u.build_train_configs("
            f"  repo_id={repo_id!r}, exp_name={args.exp_name!r},"
            f"  num_train_steps={args.max_steps}, batch_size={args.batch_size},"
            f"  save_interval={args.save_interval}, lora={args.lora}, pi05={args.pi05},"
            f"  prompt_from_task={args.prompt_from_task},"
            f"  default_prompt={args.default_prompt!r})); "
            f"sys.argv = ['train.py'] + {train_argv_tail!r}; "
            "runpy.run_path('scripts/train.py', run_name='__main__')"
        )
        subprocess.run(
            [sys.executable, "-c", bootstrap],
            cwd=str(args.openpi_dir),
            env=env,
            check=True,
        )
    else:
        _LOGGER.info("dry-run: would launch scripts/train.py with %s", train_argv_tail)

    _LOGGER.info("Done. Checkpoints under %s/checkpoints/%s/%s", args.output, cfg.name, args.exp_name)


if __name__ == "__main__":
    main()
