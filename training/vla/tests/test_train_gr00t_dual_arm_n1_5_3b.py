"""Behavior tests for the GR00T N1.5-3B dual-arm (v2.1) trainer.

Covers the parser defaults (including the ``--dataset-manifest`` mixture flag)
and the pure ``_load_dataset_specs`` resolver that turns CLI args into the list
of dataset roots to train on.
"""

from __future__ import annotations

import json

import pytest
from conftest import load_vla_module

_MOD = load_vla_module("vla_train_gr00t_n1_5_3b", "train_gr00t_dual_arm_n1_5_3b.py")


class TestCreateParser:
    def test_defaults(self):
        args = _MOD.create_parser().parse_args([])
        assert args.max_steps == 5000
        assert args.batch_size == 16
        assert args.save_steps == 500
        assert args.video_backend == "decord"
        assert args.dataset_manifest is None
        assert args.smoke_test is False

    def test_manifest_arg(self, tmp_path):
        manifest = tmp_path / "m.json"
        args = _MOD.create_parser().parse_args(["--dataset-manifest", str(manifest)])
        assert args.dataset_manifest == manifest


class TestLoadDatasetSpecs:
    def test_single_dataset_fallback(self):
        args = _MOD.create_parser().parse_args(["--dataset", "/data/run", "--video-backend", "torchcodec"])
        specs = _MOD._load_dataset_specs(args)
        assert specs == [{"path": str(args.dataset), "video_backend": "torchcodec"}]

    def test_manifest_expands_to_multiple_specs(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "datasets": [
                        {"path": "/data/a"},
                        {"path": "/data/b", "video_backend": "torchcodec"},
                    ]
                }
            )
        )
        args = _MOD.create_parser().parse_args(["--dataset-manifest", str(manifest)])
        specs = _MOD._load_dataset_specs(args)

        assert specs == [
            {"path": "/data/a", "video_backend": "decord"},
            {"path": "/data/b", "video_backend": "torchcodec"},
        ]

    def test_empty_manifest_raises(self, tmp_path):
        manifest = tmp_path / "empty.json"
        manifest.write_text(json.dumps({"datasets": []}))
        args = _MOD.create_parser().parse_args(["--dataset-manifest", str(manifest)])
        with pytest.raises(ValueError, match="no datasets"):
            _MOD._load_dataset_specs(args)
