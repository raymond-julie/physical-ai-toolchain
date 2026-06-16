#!/usr/bin/env python3
"""Fine-tune openpi (pi0 / pi0.5) on the UR5e dual-arm LeRobot dataset.

Sibling of ``train_openpi_ur10e.py``; uses the dual-arm policy/data config in
``openpi_ur5e_dual_arm_policy.py``. Runs from inside a checked-out openpi repo
at ``--openpi-dir`` (default ``/opt/openpi``).

Flow:
    1. Validate the LeRobot dataset at $DATASET_PATH (expects 14-DoF
       state/action and four ``observation.images.color_*`` cameras).
    2. Programmatically build & register a TrainConfig.
    3. Compute normalization stats.
    4. Launch ``scripts/train.py`` via JAX.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET = Path(os.environ.get("DATASET_PATH", "/data"))
DEFAULT_OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/outputs/openpi-ur5e-dual"))
DEFAULT_OPENPI_DIR = Path(os.environ.get("OPENPI_DIR", "/opt/openpi"))
DEFAULT_EXP_NAME = os.environ.get("EXP_NAME", "ur5e_dual_run")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--openpi-dir", type=Path, default=DEFAULT_OPENPI_DIR)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--repo-id", default=None,
                        help="LeRobot repo id; defaults to the dataset directory path.")
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--pi0", dest="pi05", action="store_false")
    parser.add_argument("--lora", action="store_true",
                        help="LoRA fine-tune (~22 GB VRAM); else full FT (~70 GB).")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=25_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--default-prompt", default=None,
                        help="Override prompt when tasks.jsonl is generic/missing.")
    parser.add_argument("--prompt-from-task", action="store_true", default=True)
    parser.add_argument("--use-secondary-base", action="store_true",
                        help="Use observation.images.color_1 as the base view "
                             "instead of color_0.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_dataset(path: Path) -> dict:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"meta/info.json not found under {path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    required = [
        "observation.state",
        "action",
        "observation.images.color_0",
        "observation.images.color_2",
        "observation.images.color_3",
    ]
    missing = [k for k in required if k not in features]
    if missing:
        sys.exit(f"Dataset {path} missing required features: {missing}")
    state_shape = features["observation.state"]["shape"]
    if state_shape != [14]:
        sys.exit(f"Expected state shape [14], got {state_shape}")
    _LOGGER.info(
        "Dataset OK: episodes=%s frames=%s fps=%s cameras=%s",
        info.get("total_episodes"),
        info.get("total_frames"),
        info.get("fps"),
        sorted(k for k in features if k.startswith("observation.images.")),
    )
    return info


def _bootstrap_snippet(args: argparse.Namespace, repo_id: str) -> str:
    """Python code that registers the TrainConfig in a child process."""
    here = str(Path(__file__).resolve().parent)
    openpi_src = str(args.openpi_dir / "src")
    return (
        f"import sys; "
        f"sys.path.insert(0, {here!r}); "
        f"sys.path.insert(0, {openpi_src!r}); "
        "import openpi_ur5e_dual_arm_policy as u; "
        "u.register(u.build_train_configs("
        f"  repo_id={repo_id!r}, exp_name={args.exp_name!r},"
        f"  num_train_steps={args.max_steps}, batch_size={args.batch_size},"
        f"  save_interval={args.save_interval}, lora={args.lora}, pi05={args.pi05},"
        f"  prompt_from_task={args.prompt_from_task},"
        f"  default_prompt={args.default_prompt!r},"
        f"  use_secondary_base={args.use_secondary_base},"
        f"  assets_base_dir={str(args.output / 'assets')!r},"
        f"  checkpoint_base_dir={str(args.output / 'checkpoints')!r})); "
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = create_parser().parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.openpi_dir = args.openpi_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    if not (args.openpi_dir / "scripts" / "train.py").exists():
        sys.exit(f"openpi clone missing scripts/train.py at {args.openpi_dir}")

    _validate_dataset(args.dataset)
    repo_id = args.repo_id or str(args.dataset)

    # Probe the config name so we can echo it in logs (lora and pi05 toggles
    # change the registered name).
    if args.pi05:
        cfg_name = "pi05_ur5e_dual_lora" if args.lora else "pi05_ur5e_dual"
    else:
        cfg_name = "pi0_ur5e_dual_lora" if args.lora else "pi0_ur5e_dual"

    common_args = [
        f"--assets-base-dir={args.output / 'assets'}",
        f"--checkpoint-base-dir={args.output / 'checkpoints'}",
        f"--exp-name={args.exp_name}",
    ]

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    env.setdefault("HF_HOME", str(args.output / "hf-cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(args.output / "hf-cache"))
    extra_py = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = extra_py + os.pathsep + env.get("PYTHONPATH", "")

    boot = _bootstrap_snippet(args, repo_id=repo_id)

    if not args.skip_norm_stats:
        _LOGGER.info("--- Computing norm stats for config: %s ---", cfg_name)
        norm_snippet = boot + (
            f"sys.argv = ['compute_norm_stats.py', '--config-name={cfg_name}']; "
            "import runpy; runpy.run_path('scripts/compute_norm_stats.py', run_name='__main__')"
        )
        if not args.dry_run:
            subprocess.run(
                [sys.executable, "-c", norm_snippet],
                cwd=str(args.openpi_dir), env=env, check=True,
            )
    else:
        _LOGGER.info("Skipping compute_norm_stats")

    _LOGGER.info("--- Launching openpi training: %s ---", cfg_name)
    train_argv_tail = [cfg_name, *common_args]
    if args.fsdp_devices and args.fsdp_devices > 1:
        train_argv_tail.append(f"--fsdp-devices={args.fsdp_devices}")
    if args.overwrite:
        train_argv_tail.append("--overwrite")
    if args.resume:
        train_argv_tail.append("--resume")

    train_snippet = boot + (
        f"sys.argv = ['train.py'] + {train_argv_tail!r}; "
        "import runpy; runpy.run_path('scripts/train.py', run_name='__main__')"
    )
    if not args.dry_run:
        subprocess.run(
            [sys.executable, "-c", train_snippet],
            cwd=str(args.openpi_dir), env=env, check=True,
        )
    else:
        _LOGGER.info("dry-run: train.py %s", train_argv_tail)

    _LOGGER.info("Done. Checkpoints under %s/checkpoints/%s/%s", args.output, cfg_name, args.exp_name)


if __name__ == "__main__":
    main()
