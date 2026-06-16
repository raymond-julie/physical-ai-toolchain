"""Tests for the MCAP -> LeRobot v2.1 converter pure logic (no MCAP/ffmpeg)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# lerobot-converter is hyphenated so the converter cannot be imported as a normal
# package. Load it by file path; only numpy is needed at import time (av, mcap,
# mcap_ros2, PIL, pyarrow, and tqdm are imported lazily inside functions).
_SUT_PATH = Path(__file__).resolve().parent.parent / "lerobot-converter" / "convert_mcap_to_lerobot.py"
_spec = importlib.util.spec_from_file_location("convert_mcap_to_lerobot", _SUT_PATH)
converter = importlib.util.module_from_spec(_spec)
sys.modules["convert_mcap_to_lerobot"] = converter
_spec.loader.exec_module(converter)


def _time_series(times, values):
    series = converter.TimeSeries()
    series.times = list(times)
    series.values = [np.asarray(v, dtype=np.float32) for v in values]
    series.finalize()
    return series


def _image_series(times):
    series = converter.ImageSeries()
    series.times = list(times)
    series.jpegs = [b"" for _ in times]
    series.order()
    return series


class TestTimeSeriesResample:
    """Nearest-neighbor resampling onto a uniform timeline."""

    def test_sample_nearest_picks_closest(self):
        series = _time_series([0, 10, 20], [[0.0, 100.0], [10.0, 110.0], [20.0, 120.0]])
        out = series.sample_nearest(np.array([4, 6, 15], dtype=np.int64))
        # 4 -> nearest 0; 6 -> nearest 10; 15 -> tie resolves to the left (10).
        np.testing.assert_allclose(out, [[0.0, 100.0], [10.0, 110.0], [10.0, 110.0]])

    def test_sample_nearest_clamps_out_of_range(self):
        series = _time_series([100, 200], [[1.0], [2.0]])
        out = series.sample_nearest(np.array([50, 250], dtype=np.int64))
        np.testing.assert_allclose(out, [[1.0], [2.0]])

    def test_sample_nearest_exact_timestamps(self):
        series = _time_series([0, 10, 20], [[0.0], [1.0], [2.0]])
        out = series.sample_nearest(np.array([0, 10, 20], dtype=np.int64))
        np.testing.assert_allclose(out, [[0.0], [1.0], [2.0]])


class TestImageSeriesResample:
    """Nearest-neighbor frame index selection."""

    def test_nearest_indices(self):
        series = _image_series([0, 10, 20])
        idx = series.nearest_indices(np.array([4, 6, 15], dtype=np.int64))
        np.testing.assert_array_equal(idx, [0, 1, 1])

    def test_nearest_indices_clamps(self):
        series = _image_series([100, 200])
        idx = series.nearest_indices(np.array([50, 250], dtype=np.int64))
        np.testing.assert_array_equal(idx, [0, 1])


class TestBuildGrid:
    """Uniform timeline construction over the overlap of all streams."""

    def test_grid_length_and_start(self):
        times = [0, 500_000_000, 1_000_000_000]
        grid, t0 = converter.build_grid(
            {"s": _time_series(times, [[0.0]] * 3)},
            {"a": _time_series(times, [[0.0]] * 3)},
            {"w": _time_series(times, [[0.0]] * 3)},
            {"i": _image_series(times)},
            fps=10,
        )
        assert t0 == 0
        assert len(grid) == 10
        assert grid[0] == 0

    def test_grid_uses_overlap_window(self):
        grid, t0 = converter.build_grid(
            {"s": _time_series([0, 1_000_000_000], [[0.0], [0.0]])},
            {"a": _time_series([200_000_000, 1_200_000_000], [[0.0], [0.0]])},
            {"w": _time_series([100_000_000, 1_100_000_000], [[0.0], [0.0]])},
            {"i": _image_series([0, 1_000_000_000])},
            fps=5,
        )
        # Overlap window is [max start, min end] = [200ms, 1000ms] -> 0.8s -> 4 frames.
        assert t0 == 200_000_000
        assert grid[0] == 200_000_000
        assert len(grid) == 4


class TestAssembleState:
    """Per-topic nearest samples concatenated into one state/action array."""

    def test_concatenates_topic_columns(self):
        series_map = {
            "t1": _time_series([0, 10], [[1.0, 2.0], [3.0, 4.0]]),
            "t2": _time_series([0, 10], [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        }
        topics = [("t1", "position"), ("t2", "position")]
        out = converter.assemble_state(series_map, topics, np.array([0, 10], dtype=np.int64))
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 5.0, 6.0, 7.0])
        np.testing.assert_allclose(out[1], [3.0, 4.0, 8.0, 9.0, 10.0])


class TestFeatureStats:
    """min/max/mean/std/count assembly for parquet feature columns."""

    def test_stats_values(self):
        arr = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        stats = converter.feature_stats(arr)
        assert stats["min"] == [0.0, 2.0]
        assert stats["max"] == [2.0, 4.0]
        assert stats["mean"] == [1.0, 3.0]
        assert stats["count"] == [2]


class TestImageStats:
    """Per-channel normalized image statistics shaped (3, 1, 1)."""

    def test_shapes_and_normalization(self):
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        frames[..., 0] = 255  # constant red channel
        stats = converter.image_stats(frames)
        assert np.asarray(stats["max"]).shape == (3, 1, 1)
        assert stats["max"][0] == [[1.0]]
        assert stats["min"][0] == [[1.0]]
        assert stats["max"][1] == [[0.0]]
        assert stats["count"] == [4]


class TestBuildFeatures:
    """info.json feature schema assembly for the bimanual UR embodiment."""

    def test_feature_schema(self):
        features = converter.build_features(480, 640, 15)
        assert features["observation.state"]["shape"] == [converter.STATE_DIM]
        assert features["action"]["shape"] == [converter.STATE_DIM]
        assert features["observation.state"]["names"]["motors"] == converter.STATE_ACTION_NAMES
        assert features["observation.tcp_wrench.left"]["shape"] == [converter.WRENCH_DIM]
        for cam in converter.CAMERAS:
            assert features[cam]["dtype"] == "video"
            assert features[cam]["shape"] == [480, 640, 3]
            assert features[cam]["info"]["video.fps"] == 15
        for column in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            assert column in features
