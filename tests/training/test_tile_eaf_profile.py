from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.tile_eaf_profile import build_tile_eaf_profile, summarize_tile_eaf_timing


def test_timing_summary_keeps_unattributed_wall_time() -> None:
    result = summarize_tile_eaf_timing(
        {
            "elapsed_seconds": 10.0,
            "tiles": 500,
            "phase_seconds": {"data_wait": 6.0, "teacher": 2.0},
        }
    )
    assert result["tiles_per_second"] == 50.0
    assert result["bottleneck"] == "data_wait"
    assert result["phase_share"]["data_wait"] == 0.6
    assert result["unattributed_seconds"] == 2.0


def test_profile_recommends_io_tuning_when_data_wait_dominates() -> None:
    profile = build_tile_eaf_profile(
        [
            {
                "epoch": 1,
                "train": {
                    "elapsed_seconds": 10.0,
                    "tiles": 500,
                    "phase_seconds": {"data_wait": 7.0, "teacher": 2.0},
                },
                "val": {
                    "elapsed_seconds": 2.0,
                    "tiles": 100,
                    "phase_seconds": {"teacher": 1.0},
                },
            }
        ]
    )
    assert profile["format"] == "tile_eaf_profile_v2"
    assert profile["aggregate"]["train"]["bottleneck"] == "data_wait"
    assert profile["recommendation"].startswith("I/O-bound")
