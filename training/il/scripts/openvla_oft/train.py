"""OpenVLA-OFT fine-tuning wrapper with Azure ML MLflow integration.

Mirrors ``training/il/scripts/gr00t/train.py``. OFT's upstream
``vla-scripts/finetune.py`` logs metrics directly through the W&B SDK
(``wandb.log`` and ``wandb.init`` — there are no stdout prints). We hijack
that path by prepending the wandb-shim package
(``training/il/scripts/mlflow_shim/``) to ``PYTHONPATH`` so the
subprocess imports our forwarder; every numeric metric is then re-emitted
via ``mlflow.log_metrics``.

Usage::

    python -m training.il.scripts.openvla_oft.train torchrun --standalone \
        --nnodes 1 --nproc-per-node 1 \
        vla-scripts/finetune.py [finetune args...]
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

SYSTEM_METRICS_INTERVAL = 30.0


def _shim_path() -> str:
    """Absolute path to the mlflow shim package directory."""
    return str(Path(__file__).resolve().parent.parent / "mlflow_shim")


def _init_system_collector() -> Any | None:
    """Initialize the SystemMetricsCollector when ``SYSTEM_METRICS`` is enabled."""
    if os.environ.get("SYSTEM_METRICS", "true").lower() != "true":
        return None
    try:
        from training.utils.metrics import SystemMetricsCollector

        collector = SystemMetricsCollector(collect_gpu=True, collect_disk=True)
        print("[mlflow] system metrics collection initialized")
        return collector
    except ImportError as exc:
        print(f"[mlflow] system metrics unavailable: {exc}", file=sys.stderr)
        return None


def _lineage_tags() -> dict[str, str]:
    """Build framework / dataset / VLA-path lineage tags from environment."""
    tags: dict[str, str] = {
        "framework": "openvla-oft",
        "framework.version": os.environ.get("OPENVLA_OFT_REF", "main"),
        "framework.base_model": os.environ.get("VLA_PATH", "openvla/openvla-7b"),
    }
    dataset_name = os.environ.get("DATASET_NAME", "")
    if dataset_name:
        tags["dataset.name"] = dataset_name
    dataset_mount = os.environ.get("DATASET_MOUNT", "")
    if dataset_mount and "${{" not in dataset_mount:
        tags["dataset.mount"] = dataset_mount
    for env_key, tag_key in (
        ("BATCH_SIZE", "hyperparams.batch_size"),
        ("LEARNING_RATE", "hyperparams.learning_rate"),
        ("MAX_STEPS", "hyperparams.max_steps"),
        ("NUM_GPUS", "hyperparams.num_gpus"),
        ("LORA_RANK", "hyperparams.lora_rank"),
        ("NUM_ACTIONS_CHUNK", "hyperparams.num_actions_chunk"),
        ("USE_FILM", "hyperparams.use_film"),
        ("USE_PROPRIO", "hyperparams.use_proprio"),
        ("USE_L1_REGRESSION", "hyperparams.use_l1_regression"),
        ("NUM_IMAGES_IN_INPUT", "hyperparams.num_images_in_input"),
    ):
        value = os.environ.get(env_key, "")
        if value:
            tags[tag_key] = value
    return tags


def _system_metrics_loop(
    collector: Any,
    mlflow: Any,
    stop_event: threading.Event,
) -> None:
    """Background thread that logs system metrics every SYSTEM_METRICS_INTERVAL."""
    while not stop_event.is_set():
        try:
            metrics = collector.collect_metrics()
        except Exception as exc:
            print(f"[mlflow] system metrics collection failed: {exc}", file=sys.stderr)
            metrics = {}
        if metrics:
            with contextlib.suppress(Exception):
                mlflow.log_metrics(metrics)
        stop_event.wait(SYSTEM_METRICS_INTERVAL)


def _bootstrap_run() -> Any | None:
    """Attach to the AzureML-managed MLflow run, or start a local one."""
    try:
        import mlflow
    except ImportError:
        print("[mlflow] mlflow not installed; metrics will not be logged", file=sys.stderr)
        return None

    if os.environ.get("MLFLOW_RUN_ID"):
        print(f"[mlflow] attaching to AzureML-managed run {os.environ['MLFLOW_RUN_ID']}")
        try:
            if mlflow.active_run() is None:
                mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"], nested=False)
        except Exception as exc:
            print(f"[mlflow] failed to attach to managed run: {exc}", file=sys.stderr)
            return None
        return mlflow

    try:
        from training.il.scripts.lerobot.bootstrap import bootstrap_mlflow

        bootstrap_mlflow(
            experiment_name=os.environ.get("EXPERIMENT_NAME", "openvla-oft-training"),
            policy_type="openvla-oft",
            job_name=os.environ.get("JOB_NAME", "openvla-oft-training"),
        )
        return mlflow
    except SystemExit:
        print("[mlflow] no Azure ML context; running without MLflow logging", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[mlflow] bootstrap failed: {exc}", file=sys.stderr)
        return None


def _stream_output(process: subprocess.Popen[str]) -> None:
    """Forward subprocess stdout to our stdout line-by-line."""
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()


def run_training(cmd: list[str]) -> int:
    """Execute the torchrun command with MLflow monitoring.

    Args:
        cmd: Full command (torchrun + finetune.py + args).

    Returns:
        Subprocess exit code.
    """
    mlflow = _bootstrap_run()

    if mlflow is not None:
        try:
            mlflow.set_tags(_lineage_tags())
        except Exception as exc:
            print(f"[mlflow] set_tags failed: {exc}", file=sys.stderr)

    env = os.environ.copy()
    shim = _shim_path()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = shim + (os.pathsep + existing if existing else "")
    # Prevent the real W&B client (transitive dep) from doing any network I/O
    # even though our shim shadows the import on PYTHONPATH (defense in depth).
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    print(f"[mlflow] shim mounted at {shim} (intercepts upstream wandb calls)")

    print(f"[mlflow] launching: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def _forward(signum: int, _frame: object) -> None:
        print(f"[mlflow] received signal {signum}, terminating training subprocess")
        process.terminate()

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    stop_event = threading.Event()
    metrics_thread: threading.Thread | None = None
    if mlflow is not None:
        collector = _init_system_collector()
        if collector is not None:
            metrics_thread = threading.Thread(
                target=_system_metrics_loop,
                args=(collector, mlflow, stop_event),
                daemon=True,
            )
            metrics_thread.start()

    try:
        _stream_output(process)
    finally:
        process.wait()
        stop_event.set()
        if metrics_thread is not None:
            metrics_thread.join(timeout=2.0)

    return_code = process.returncode or EXIT_SUCCESS
    if mlflow is not None:
        with contextlib.suppress(Exception):
            mlflow.set_tag(
                "training_status",
                "completed" if return_code == EXIT_SUCCESS else "failed",
            )

    return return_code


def main() -> int:
    """Entry point. ``sys.argv[1:]`` must be the full subprocess command."""
    if len(sys.argv) < 2:
        print(
            "usage: python -m training.il.scripts.openvla_oft.train <torchrun cmd...>",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    start = time.time()
    exit_code = run_training(sys.argv[1:])
    elapsed = time.time() - start
    print(f"[mlflow] training finished in {elapsed:.1f}s with exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
