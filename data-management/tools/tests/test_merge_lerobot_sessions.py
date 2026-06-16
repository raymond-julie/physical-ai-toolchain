"""Tests for LeRobot session merge re-indexing logic (no pyarrow/datasets)."""

from __future__ import annotations

import pytest
from merge_lerobot_sessions import (
    MergeError,
    build_combined_info,
    discover_sessions,
    discover_video_keys,
    frame_index_range,
    reindex_episode_entry,
)


class TestDiscoverSessions:
    """Session discovery returns only sorted ``session_*`` directories."""

    def test_returns_sorted_session_dirs(self, tmp_path):
        (tmp_path / "session_b").mkdir()
        (tmp_path / "session_a").mkdir()
        (tmp_path / "not_a_session").mkdir()
        (tmp_path / "session_c.txt").write_text("x")
        sessions = discover_sessions(tmp_path)
        assert [p.name for p in sessions] == ["session_a", "session_b"]

    def test_raises_when_no_sessions(self, tmp_path):
        with pytest.raises(MergeError):
            discover_sessions(tmp_path)


class TestDiscoverVideoKeys:
    """Video keys are the features whose dtype is ``video``."""

    def test_filters_video_dtype(self):
        info = {
            "features": {
                "observation.state": {"dtype": "float32"},
                "observation.images.color": {"dtype": "video"},
                "observation.images.color2": {"dtype": "video"},
            }
        }
        assert discover_video_keys(info) == ["observation.images.color", "observation.images.color2"]


class TestReindexEpisodeEntry:
    """Episode entries are renumbered without mutating the source."""

    def test_reindex_preserves_fields(self):
        out = reindex_episode_entry({"episode_index": 0, "tasks": ["pick"], "length": 120}, 7)
        assert out["episode_index"] == 7
        assert out["tasks"] == ["pick"]
        assert out["length"] == 120

    def test_reindex_does_not_mutate_original(self):
        entry = {"episode_index": 0, "tasks": ["pick"], "length": 120}
        reindex_episode_entry(entry, 7)
        assert entry["episode_index"] == 0


class TestFrameIndexRange:
    """Per-episode global frame indices are contiguous."""

    def test_contiguous_range(self):
        assert frame_index_range(100, 3) == [100, 101, 102]

    def test_empty_episode(self):
        assert frame_index_range(50, 0) == []


class TestGlobalReindexing:
    """Episodes and frames renumber sequentially across merged sessions."""

    def test_two_session_renumbering(self):
        sessions = [
            [
                {"episode_index": 0, "tasks": ["a"], "length": 3},
                {"episode_index": 1, "tasks": ["a"], "length": 2},
            ],
            [{"episode_index": 0, "tasks": ["b"], "length": 4}],
        ]
        merged = []
        frame_ranges = []
        global_ep = 0
        global_frame = 0
        for session in sessions:
            for entry in session:
                merged.append(reindex_episode_entry(entry, global_ep))
                frame_ranges.append(frame_index_range(global_frame, entry["length"]))
                global_frame += entry["length"]
                global_ep += 1

        assert [m["episode_index"] for m in merged] == [0, 1, 2]
        assert [m["tasks"] for m in merged] == [["a"], ["a"], ["b"]]
        flat = [index for rng in frame_ranges for index in rng]
        assert flat == list(range(3 + 2 + 4))
        assert frame_ranges[2] == [5, 6, 7, 8]


class TestBuildCombinedInfo:
    """Combined info carries merged totals and a refreshed train split."""

    def test_updates_totals_and_splits(self):
        ref = {
            "codebase_version": "v2.1",
            "total_episodes": 1,
            "total_frames": 10,
            "total_videos": 2,
            "splits": {"train": "0:1"},
            "features": {},
        }
        combined = build_combined_info(ref, 5, 100, 10)
        assert combined["total_episodes"] == 5
        assert combined["total_frames"] == 100
        assert combined["total_videos"] == 10
        assert combined["splits"] == {"train": "0:5"}

    def test_does_not_mutate_reference(self):
        ref = {"total_episodes": 1, "total_frames": 10, "total_videos": 2, "splits": {"train": "0:1"}}
        build_combined_info(ref, 5, 100, 10)
        assert ref["total_episodes"] == 1
