"""Behavior tests for the GR00T dual-arm trainer CLI and entry point.

Exercises the argument parser defaults and the CPU-only ``--smoke-test`` exit
paths. ``torch`` is stubbed (``cuda.is_available() -> False``) by the package
conftest, so ``main`` takes its CUDA-absent branch without importing GR00T.
"""

from __future__ import annotations

import sys

import pytest
from conftest import load_vla_module

_MOD = load_vla_module("vla_train_gr00t_dual_arm", "train_gr00t_dual_arm.py")


class TestCreateParser:
    def test_defaults(self):
        args = _MOD.create_parser().parse_args([])
        assert args.max_steps == 5000
        assert args.batch_size == 16
        assert args.save_steps == 25000
        assert args.grad_accum == 2
        assert args.lr == pytest.approx(1e-4)
        assert args.video_backend == "decord"
        assert args.smoke_test is False
        assert args.resume is None

    def test_smoke_test_flag(self):
        args = _MOD.create_parser().parse_args(["--smoke-test"])
        assert args.smoke_test is True

    def test_overrides(self):
        args = _MOD.create_parser().parse_args(
            ["--max-steps", "10", "--batch-size", "2", "--video-backend", "torchcodec"]
        )
        assert args.max_steps == 10
        assert args.batch_size == 2
        assert args.video_backend == "torchcodec"

    def test_invalid_video_backend_rejected(self):
        with pytest.raises(SystemExit):
            _MOD.create_parser().parse_args(["--video-backend", "ffmpeg"])


class TestMain:
    def test_smoke_test_exits_zero_without_cuda(self):
        # torch stub reports cuda unavailable -> smoke test returns 0 cleanly.
        assert _MOD.main(["--smoke-test"]) == 0

    def test_without_smoke_test_errors_without_cuda(self):
        assert _MOD.main([]) == 1

    def test_returns_one_when_torch_missing(self, monkeypatch):
        # A None entry makes ``import torch`` raise ImportError.
        monkeypatch.setitem(sys.modules, "torch", None)
        assert _MOD.main(["--smoke-test"]) == 1
