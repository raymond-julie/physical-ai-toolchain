"""Behavior tests for the openpi UR5e dual-arm trainer driver.

Covers the parser (pi0/pi0.5 + LoRA toggles), the ``_validate_dataset`` shape
guard against ``meta/info.json``, and the ``_bootstrap_snippet`` builder that
registers the TrainConfig in the child training process.
"""

from __future__ import annotations

import json

import pytest
from conftest import load_vla_module

_MOD = load_vla_module("vla_train_openpi_ur5e_dual_arm", "train_openpi_ur5e_dual_arm.py")


def _write_info(root, *, features: dict, episodes: int = 10) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": 100, "fps": 15, "features": features})
    )


_VALID_FEATURES = {
    "observation.state": {"shape": [14]},
    "action": {"shape": [14]},
    "observation.images.color_0": {"shape": [3, 224, 224]},
    "observation.images.color_2": {"shape": [3, 224, 224]},
    "observation.images.color_3": {"shape": [3, 224, 224]},
}


class TestCreateParser:
    def test_defaults(self):
        args = _MOD.create_parser().parse_args([])
        assert args.pi05 is True
        assert args.lora is False
        assert args.max_steps == 30_000
        assert args.batch_size == 32
        assert args.prompt_from_task is True

    def test_pi0_toggle(self):
        args = _MOD.create_parser().parse_args(["--pi0"])
        assert args.pi05 is False

    def test_lora_toggle(self):
        args = _MOD.create_parser().parse_args(["--lora"])
        assert args.lora is True


class TestValidateDataset:
    def test_valid_dataset_returns_info(self, tmp_path):
        _write_info(tmp_path, features=_VALID_FEATURES, episodes=42)
        info = _MOD._validate_dataset(tmp_path)
        assert info["total_episodes"] == 42

    def test_missing_info_json_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _MOD._validate_dataset(tmp_path)

    def test_missing_feature_exits(self, tmp_path):
        features = {k: v for k, v in _VALID_FEATURES.items() if k != "observation.images.color_3"}
        _write_info(tmp_path, features=features)
        with pytest.raises(SystemExit):
            _MOD._validate_dataset(tmp_path)

    def test_wrong_state_shape_exits(self, tmp_path):
        features = {**_VALID_FEATURES, "observation.state": {"shape": [7]}}
        _write_info(tmp_path, features=features)
        with pytest.raises(SystemExit):
            _MOD._validate_dataset(tmp_path)


class TestBootstrapSnippet:
    def test_snippet_embeds_repo_and_registration(self):
        args = _MOD.create_parser().parse_args(["--exp-name", "demo_run"])
        snippet = _MOD._bootstrap_snippet(args, repo_id="/data/combined")
        assert "/data/combined" in snippet
        assert "demo_run" in snippet
        assert "build_train_configs" in snippet
        assert "u.register" in snippet
