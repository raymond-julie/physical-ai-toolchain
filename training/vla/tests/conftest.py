"""Pytest setup for the training/vla scripts.

Installs lightweight ``sys.modules`` stubs for the heavy / GPU-only dependencies
(``torch``, ``transformers``, ``gr00t``, ``jax``, ``openpi``, ``lerobot``, ...)
BEFORE any script-under-test is imported, then exposes ``load_vla_module`` to
load a script from ``training/vla/scripts/`` by file path. This keeps the suite
runnable on a CPU-only host with no model frameworks installed.

The trainers import their heavy frameworks lazily (inside functions or under
``TYPE_CHECKING``), so they need no real packages to import. The openpi policy
module, however, subclasses ``openpi`` base classes at import time, so the
``openpi`` stub below provides real (not mock) base classes and config holders.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_vla_module(name: str, filename: str) -> types.ModuleType:
    """Load a ``training/vla/scripts`` module by file path under ``name``.

    Loading by path (rather than importing a package) sidesteps the hyphen-free
    but collision-prone ``scripts`` package name and lets each test pick a unique
    module name.
    """
    full_path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, full_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name!r} from {full_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _register(name: str, module: types.ModuleType) -> types.ModuleType:
    """Register ``module`` in ``sys.modules`` and attach it to its parent."""
    sys.modules[name] = module
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        setattr(sys.modules[parent_name], child, module)
    return module


def _install_framework_stubs() -> None:
    """Register placeholder modules for the GPU / model frameworks."""
    # torch needs a real ``cuda.is_available`` returning a falsey bool so the
    # trainers take their CUDA-absent branch instead of importing GR00T.
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)

    for name in ("transformers", "jax", "lerobot", "accelerate", "decord", "gr00t"):
        sys.modules.setdefault(name, MagicMock(name=name))

    einops = types.ModuleType("einops")

    def _rearrange(tensor: Any, pattern: str, **axes: int) -> Any:
        import numpy as np

        if pattern.replace(" ", "") == "chw->hwc":
            return np.transpose(tensor, (1, 2, 0))
        raise NotImplementedError(pattern)

    einops.rearrange = _rearrange
    sys.modules.setdefault("einops", einops)


def _install_openpi_stub() -> None:
    """Register a minimal but functional ``openpi`` package tree.

    The base classes that ``openpi_ur5e_dual_arm_policy`` subclasses are real
    classes; the config holders store their kwargs so tests can assert on the
    assembled ``TrainConfig``.
    """
    if "openpi" in sys.modules:
        return

    _register("openpi", types.ModuleType("openpi"))

    transforms = _register("openpi.transforms", types.ModuleType("openpi.transforms"))

    class _DataTransformFn:
        """Base for the dual-arm input/output transforms (subclassed at import)."""

    transforms.DataTransformFn = _DataTransformFn
    transforms.Group = MagicMock(name="Group")
    transforms.RepackTransform = MagicMock(name="RepackTransform")
    transforms.DeltaActions = MagicMock(name="DeltaActions")
    transforms.AbsoluteActions = MagicMock(name="AbsoluteActions")
    transforms.make_bool_mask = lambda *args: tuple(args)

    _register("openpi.models", types.ModuleType("openpi.models"))
    model_mod = _register("openpi.models.model", types.ModuleType("openpi.models.model"))
    model_mod.ModelType = type("ModelType", (), {})
    model_mod.BaseModelConfig = type("BaseModelConfig", (), {})

    pi0_mod = _register("openpi.models.pi0_config", types.ModuleType("openpi.models.pi0_config"))

    class _Pi0Config:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def get_freeze_filter(self) -> str:
            return "freeze-filter"

    pi0_mod.Pi0Config = _Pi0Config

    _register("openpi.training", types.ModuleType("openpi.training"))
    config_mod = _register("openpi.training.config", types.ModuleType("openpi.training.config"))

    @dataclasses.dataclass(frozen=True)
    class _DataConfigFactory:
        repo_id: str = ""
        base_config: Any = None

        def create_base_config(self, assets_dirs: Any, model_config: Any) -> Any:
            return types.SimpleNamespace()

    class _DataConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class _ModelTransformFactory:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, model_config: Any) -> Any:
            return MagicMock(name="model_transforms")

    class _TrainConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    config_mod.DataConfigFactory = _DataConfigFactory
    config_mod.DataConfig = _DataConfig
    config_mod.ModelTransformFactory = _ModelTransformFactory
    config_mod.TrainConfig = _TrainConfig
    config_mod._CONFIGS_DICT = {}
    config_mod._CONFIGS = []

    weight_mod = _register("openpi.training.weight_loaders", types.ModuleType("openpi.training.weight_loaders"))

    class _CheckpointWeightLoader:
        def __init__(self, path: str) -> None:
            self.path = path

    weight_mod.CheckpointWeightLoader = _CheckpointWeightLoader


_install_framework_stubs()
_install_openpi_stub()
