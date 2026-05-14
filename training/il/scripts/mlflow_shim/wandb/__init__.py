"""wandb-compatible shim that forwards calls to MLflow.

Upstream training scripts (NVIDIA Isaac-GR00T via HuggingFace `WandbCallback`,
moojink/openvla-oft via direct `wandb.log` calls) emit metrics through the
Weights & Biases SDK. We do not use W&B — metrics belong in Azure ML MLflow.

Rather than patching upstream source code, we prepend this package's parent
directory to ``PYTHONPATH`` so ``import wandb`` resolves to this shim before
the real ``wandb`` site-package. The shim implements only the surface area
called by HF Trainer and OFT (init/log/config/watch/finish + media classes)
and routes numeric metrics to ``mlflow.log_metrics``.

The shim runs inside the training subprocess. It attaches to the AzureML-
managed MLflow run via ``MLFLOW_RUN_ID`` (auto-injected by the AzureML K8s
extension) on first ``log()`` call. Non-numeric values are dropped silently
to match the wandb API's permissive behavior.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any

__version__ = "0.0.0+mlflow-shim"

_run_lock = threading.Lock()
_run_started = False
_params_logged: set[str] = set()


def _mlflow() -> Any:
    """Import mlflow lazily so the shim loads even when mlflow is missing."""
    import mlflow

    return mlflow


def _ensure_run() -> None:
    """Attach to the AzureML-managed MLflow run on first log call."""
    global _run_started
    if _run_started:
        return
    with _run_lock:
        if _run_started:
            return
        try:
            mlflow = _mlflow()
            if mlflow.active_run() is None:
                run_id = os.environ.get("MLFLOW_RUN_ID")
                if run_id:
                    mlflow.start_run(run_id=run_id, nested=False)
                else:
                    mlflow.start_run()
            _run_started = True
        except Exception as exc:
            print(f"[mlflow-shim] failed to start mlflow run: {exc}", file=sys.stderr)


def _coerce_param(value: Any) -> Any:
    """Coerce a wandb config value to an mlflow-loggable scalar or string."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _log_params(params: dict[str, Any]) -> None:
    """Log parameters to MLflow, skipping ones already logged in this run."""
    new_params: dict[str, Any] = {}
    for key, value in params.items():
        skey = str(key)
        if skey in _params_logged:
            continue
        coerced = _coerce_param(value)
        if coerced is None:
            continue
        new_params[skey] = coerced
        _params_logged.add(skey)
    if not new_params:
        return
    try:
        _mlflow().log_params(new_params)
    except Exception as exc:
        print(f"[mlflow-shim] log_params failed: {exc}", file=sys.stderr)


class _Stub:
    """Generic stub for ``wandb.Image`` / ``Video`` / ``Histogram`` / etc."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return _Stub()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _Stub()


# wandb-API kwargs that callers pass to ``config.update`` but that are not
# user-meaningful parameters (they configure wandb's own behavior).
_WANDB_CONFIG_RESERVED_KWARGS = frozenset({"allow_val_change", "exclude", "include"})


class _Config(dict):
    """wandb.config replacement that logs every set value as an mlflow param."""

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        merged: dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            merged.update(args[0])
        # Drop wandb-reserved kwargs (e.g. ``allow_val_change=True``) so they
        # don't leak into MLflow params.
        for key, value in kwargs.items():
            if key in _WANDB_CONFIG_RESERVED_KWARGS:
                continue
            merged[key] = value
        super().update(merged)
        _ensure_run()
        _log_params(merged)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        _ensure_run()
        _log_params({key: value})

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self.__setitem__(name, value)


class _Run:
    """wandb.run replacement exposing ``id``, ``name``, ``url``, ``config``."""

    def __init__(self, run_id: str | None = None, name: str | None = None) -> None:
        self.id = run_id or os.environ.get("MLFLOW_RUN_ID", "mlflow-shim-run")
        self.name = name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "mlflow-shim-run")
        self.url = os.environ.get("MLFLOW_TRACKING_URI", "")
        self.config = _Config()
        self.summary: dict[str, Any] = {}

    def log(self, *args: Any, **kwargs: Any) -> None:
        log(*args, **kwargs)

    def watch(self, *args: Any, **kwargs: Any) -> None:
        return

    def finish(self, *args: Any, **kwargs: Any) -> None:
        finish(*args, **kwargs)

    def log_code(self, *args: Any, **kwargs: Any) -> None:
        return

    def _label(self, *args: Any, **kwargs: Any) -> None:
        """HF Trainer calls ``run._label(code="transformers_trainer")``."""
        return

    def log_artifact(self, *args: Any, **kwargs: Any) -> Any:
        """Stubbed wandb.run.log_artifact; MLflow handles artifacts separately."""
        return _Stub()


config: _Config = _Config()
run: _Run | None = None
summary: dict[str, Any] = {}


def init(*args: Any, **kwargs: Any) -> _Run:
    """wandb.init replacement; attaches to the AzureML-managed MLflow run."""
    global run
    _ensure_run()
    name = kwargs.get("name") or kwargs.get("id")
    run = _Run(run_id=kwargs.get("id"), name=name)
    cfg = kwargs.get("config")
    if isinstance(cfg, dict):
        _log_params(cfg)
    project = kwargs.get("project")
    if project:
        with contextlib.suppress(Exception):
            _mlflow().set_tag("wandb.project", str(project))
    return run


def log(
    metrics: dict[str, Any] | None = None,
    step: int | None = None,
    commit: bool | None = None,
    **kwargs: Any,
) -> None:
    """wandb.log replacement; forwards numeric values to mlflow.log_metrics."""
    if not metrics:
        return
    _ensure_run()
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            cleaned[str(key)] = float(value)
    if not cleaned:
        return
    try:
        if step is not None:
            _mlflow().log_metrics(cleaned, step=int(step))
        else:
            _mlflow().log_metrics(cleaned)
    except Exception as exc:
        print(f"[mlflow-shim] log_metrics failed: {exc}", file=sys.stderr)


def finish(*args: Any, **kwargs: Any) -> None:
    """No-op finish — the parent wrapper closes the MLflow run."""
    return


def watch(*args: Any, **kwargs: Any) -> None:
    return


def define_metric(*args: Any, **kwargs: Any) -> None:
    return


def save(*args: Any, **kwargs: Any) -> None:
    return


def login(*args: Any, **kwargs: Any) -> bool:
    return True


def setup(*args: Any, **kwargs: Any) -> None:
    return


def termwarn(*args: Any, **kwargs: Any) -> None:
    """HF Trainer calls ``wandb.termwarn(...)`` for user-facing warnings."""
    return


def termlog(*args: Any, **kwargs: Any) -> None:
    return


def termerror(*args: Any, **kwargs: Any) -> None:
    return


def log_artifact(*args: Any, **kwargs: Any) -> Any:
    """Top-level ``wandb.log_artifact``; routed through the shim, no-op."""
    return _Stub()


Image = _Stub
Video = _Stub
Histogram = _Stub
Table = _Stub
Audio = _Stub
Object3D = _Stub
Molecule = _Stub
Html = _Stub
Plotly = _Stub
Graph = _Stub
Artifact = _Stub
