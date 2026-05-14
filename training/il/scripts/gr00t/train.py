"""GR00T N1.7 fine-tuning wrapper with Azure ML MLflow integration.

Mirrors the LeRobot pattern (``training/il/scripts/lerobot/train.py``):

* Bootstraps the Azure ML MLflow tracking URI (no-op when AzureML K8s has
  already injected ``MLFLOW_RUN_ID`` and ``MLFLOW_TRACKING_URI``).
* Prepends the wandb-shim package
  (``training/il/scripts/mlflow_shim/``) to ``PYTHONPATH`` so the
  subprocess's ``import wandb`` resolves to our MLflow forwarder rather than
  the real wandb client. HuggingFace ``Trainer.WandbCallback`` then routes
  every metric through us.
* Spawns ``torchrun gr00t/experiment/launch_finetune.py`` and streams its
  stdout to the AzureML user-log file.
* Collects host system metrics (CPU, memory, GPU, disk) on a background
  thread and logs them to MLflow every 30s.
* Tags the run with framework / dataset / base-model lineage.

The entry script is responsible for adding ``--use-wandb True`` to the
launch_finetune.py CLI so HF Trainer's ``WandbCallback`` is registered;
without it ``report_to="none"`` and no callback ever calls ``wandb.log``.

Usage::

    python -m training.il.scripts.gr00t.train torchrun --standalone \
        --nnodes 1 --nproc-per-node 1 \
        gr00t/experiment/launch_finetune.py [launch_finetune args...]
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
    """Build framework / dataset / model lineage tags from environment."""
    tags: dict[str, str] = {
        "framework": "gr00t",
        "framework.version": os.environ.get("GR00T_REF", "main"),
        "framework.base_model": os.environ.get("BASE_MODEL_PATH", "nvidia/GR00T-N1.7-3B"),
    }
    dataset_name = os.environ.get("DATASET_NAME", "")
    if dataset_name:
        tags["dataset.name"] = dataset_name
    dataset_mount = os.environ.get("DATASET_MOUNT", "")
    if dataset_mount and "${{" not in dataset_mount:
        tags["dataset.mount"] = dataset_mount
    for env_key, tag_key in (
        ("GLOBAL_BATCH_SIZE", "hyperparams.global_batch_size"),
        ("LEARNING_RATE", "hyperparams.learning_rate"),
        ("MAX_STEPS", "hyperparams.max_steps"),
        ("NUM_GPUS", "hyperparams.num_gpus"),
        ("TUNE_PROJECTOR", "hyperparams.tune_projector"),
        ("TUNE_DIFFUSION_MODEL", "hyperparams.tune_diffusion_model"),
        ("TUNE_LLM", "hyperparams.tune_llm"),
        ("TUNE_VISUAL", "hyperparams.tune_visual"),
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
    """Attach to the AzureML-managed MLflow run, or start a local one.

    Returns the imported ``mlflow`` module on success, ``None`` on failure
    (so the wrapper can fall back to running the subprocess unmonitored).
    """
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

    # Local or out-of-band invocation: try to bootstrap via the LeRobot helper
    # which resolves the workspace tracking URI from the standard Azure env vars.
    try:
        from training.il.scripts.lerobot.bootstrap import bootstrap_mlflow

        bootstrap_mlflow(
            experiment_name=os.environ.get("EXPERIMENT_NAME", "gr00t-training"),
            policy_type="gr00t",
            job_name=os.environ.get("JOB_NAME", "gr00t-training"),
        )
        return mlflow
    except SystemExit:
        # bootstrap_mlflow exits when Azure env vars are missing; that's fine
        # for purely-local runs where the user didn't ask for MLflow.
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
        cmd: Full command (torchrun + launch_finetune.py + args).

    Returns:
        Subprocess exit code.
    """
    mlflow = _bootstrap_run()

    if mlflow is not None:
        try:
            mlflow.set_tags(_lineage_tags())
        except Exception as exc:
            print(f"[mlflow] set_tags failed: {exc}", file=sys.stderr)

    # Inject the mlflow shim ahead of every other entry on PYTHONPATH so the
    # subprocess's ``import wandb`` resolves to our MLflow forwarder. Re-include
    # the parent process PYTHONPATH so ``training.*`` imports continue to resolve.
    env = os.environ.copy()
    shim = _shim_path()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = shim + (os.pathsep + existing if existing else "")
    # Even with the shim in place, prevent the real wandb client (if installed
    # in the venv as a transitive dep) from doing any network I/O.
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

    # System metrics on a background thread (only when MLflow is active).
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
            "usage: python -m training.il.scripts.gr00t.train <torchrun cmd...>",
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
