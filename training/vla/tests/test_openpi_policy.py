"""Behavior tests for the openpi UR5e dual-arm policy / data-config module.

Covers ``build_train_configs`` assembly (config name + LoRA freeze filter wiring
across the pi0 / pi0.5 toggles), ``register`` mutating openpi's config registry,
and the ``_parse_image`` normalization helper. The ``openpi`` package is stubbed
by the package conftest with real base classes; ``numpy`` is used for real.
"""

from __future__ import annotations

import sys

import numpy as np
from conftest import load_vla_module

_MOD = load_vla_module("vla_openpi_ur5e_dual_arm_policy", "openpi_ur5e_dual_arm_policy.py")
_OPENPI_CONFIG = sys.modules["openpi.training.config"]


class TestBuildTrainConfigs:
    def test_pi05_lora_naming_and_freeze_filter(self):
        config = _MOD.build_train_configs(repo_id="/data/combined", exp_name="run1", lora=True, pi05=True)
        assert config.name == "pi05_ur5e_dual_lora"
        assert config.exp_name == "run1"
        assert config.data.repo_id == "/data/combined"
        # LoRA fine-tunes wire a freeze filter and disable EMA.
        assert config.freeze_filter == "freeze-filter"
        assert config.ema_decay is None
        assert config.model.action_horizon == _MOD.DEFAULT_ACTION_HORIZON

    def test_pi0_full_finetune_naming(self):
        config = _MOD.build_train_configs(repo_id="/data/x", exp_name="run2", lora=False, pi05=False)
        assert config.name == "pi0_ur5e_dual"
        # Full fine-tunes do not set the LoRA-only freeze filter.
        assert not hasattr(config, "freeze_filter")

    def test_use_secondary_base_propagates_to_data_config(self):
        config = _MOD.build_train_configs(
            repo_id="/data/x", exp_name="run3", use_secondary_base=True
        )
        assert config.data.use_secondary_base is True


class TestRegister:
    def test_register_appends_to_openpi_registry(self):
        config = _MOD.build_train_configs(repo_id="/data/x", exp_name="reg", lora=True)
        _MOD.register(config)
        assert _OPENPI_CONFIG._CONFIGS_DICT[config.name] is config
        assert config in _OPENPI_CONFIG._CONFIGS


class TestParseImage:
    def test_float_hwc_scaled_to_uint8(self):
        image = np.ones((4, 5, 3), dtype=np.float32)
        out = _MOD._parse_image(image)
        assert out.dtype == np.uint8
        assert out.shape == (4, 5, 3)
        assert out[0, 0, 0] == 255

    def test_chw_transposed_to_hwc(self):
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        out = _MOD._parse_image(image)
        assert out.shape == (4, 5, 3)

    def test_uint8_hwc_passthrough(self):
        image = np.full((4, 5, 3), 7, dtype=np.uint8)
        out = _MOD._parse_image(image)
        assert out.dtype == np.uint8
        assert np.array_equal(out, image)
