import csv
import json
from types import SimpleNamespace

import pytest

from utra_tracker.calibrate import arena_mapping, load_calibration, save_calibration
from utra_tracker.cli import build_parser
from utra_tracker.maze import Maze, MazeError
from utra_tracker.records import Detection, result_to_detections
from utra_tracker.result import SUMMARY_COLUMNS, build_result, format_result, save_result
from utra_tracker.session import detection_to_observation


def metrics(**kw):
    m = {
        "outcome": "PASS",
        "state": "finished",
        "finish_reason": "settled in end zone",
        "elapsed_s": 42.0,
        "time_limit_s": 480.0,
        "started_in_start_zone": True,
        "start_zone_share": 1.0,
        "start_ft": [1.0, 1.0],
        "start_cell": [1, 1],
        "ended_in_end_zone": True,
        "end_zone_share": 0.9,
        "reached_end_zone_s": 40.0,
        "end_ft": [7.5, 3.5],
        "end_cell": [7, 3],
        "distance_ft": 10.2,
        "steps": 10,
        "unique_cells": 9,
        "collision_tracking": True,
        "collision_tracking_toggled": False,
        "minor_collisions": 1,
        "major_collisions": 0,
        "collisions": [],
    }
    m.update(kw)
    return m


def test_save_result_writes_json_and_appends_summary(tmp_path):
    for i, outcome in enumerate(["PASS", "FAIL"]):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        path = save_result(run_dir, build_result(str(i), "1", metrics(outcome=outcome), calibrated=True))
        assert json.loads(path.read_text())["outcome"] == outcome
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert tuple(rows[0]) == SUMMARY_COLUMNS
    assert [(r["team"], r["outcome"], r["folder"]) for r in rows] == [
        ("0", "PASS", "run0"),
        ("1", "FAIL", "run1"),
    ]
    assert rows[0]["end_cell"] == "7 3"


def test_format_result():
    text = format_result(build_result("7", "2", metrics(), calibrated=False))
    assert "Team 7  run 2: PASS" in text
    assert "1 minor, 0 major" in text
    assert "not calibrated" in text
    off = format_result(build_result("7", "2", metrics(collision_tracking=False), calibrated=True))
    assert "not tracked" in off


def test_run_flags():
    args = build_parser().parse_args(
        ["run", "--team", "7", "--run", "2", "--no-grading", "--no-collisions", "--time-limit", "60"]
    )
    assert (args.team, args.run_label, args.grading, args.collisions, args.time_limit) == (
        "7",
        "2",
        False,
        False,
        60,
    )
    for removed in (["grade", "x"], ["leaderboard"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(removed)


def test_calibration_roundtrip_and_scaling(tmp_path):
    path = tmp_path / "cal.json"
    corners = [(10.0, 20.0), (810.0, 20.0), (810.0, 420.0), (10.0, 420.0)]
    save_calibration(path, corners, (1000, 500))
    assert load_calibration(path, (1000, 500)) == corners
    assert load_calibration(path, (2000, 1000))[1] == (1620.0, 40.0)
    assert load_calibration(tmp_path / "missing.json", (1, 1)) is None
    h, calibrated = arena_mapping(Maze(), path, (1000, 500))
    assert calibrated and h.apply((810.0, 420.0)) == pytest.approx((8.0, 4.0))
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(MazeError):
        load_calibration(path, (1000, 500))


def test_uncalibrated_mapping_uses_whole_frame(tmp_path):
    h, calibrated = arena_mapping(Maze(), tmp_path / "none.json", (800, 400))
    assert not calibrated
    assert h.apply((400.0, 200.0)) == pytest.approx((4.0, 2.0))


def test_detection_to_observation_prefers_mask():
    h, _ = arena_mapping(Maze(), "missing.json", (800, 400))
    poly = ((300.0, 100.0), (340.0, 100.0), (340.0, 140.0), (300.0, 140.0))
    det = Detection(0, 0.0, 1, "robot", 0.9, 280.0, 80.0, 360.0, 160.0, polygon=poly)
    o = detection_to_observation(det, h, "mask")
    assert o.center == pytest.approx((3.2, 1.2))
    assert len(o.footprint) == 4 and o.footprint[0] == pytest.approx((3.0, 1.0))
    box = detection_to_observation(det, h, "box")
    assert box.footprint[0] == pytest.approx((2.8, 0.8))
    corrected = detection_to_observation(det, h, "mask", ((4.0, 2.0), 8.0, 0.5))
    assert corrected.center[0] > o.center[0]  # pulled toward the nadir at x=4


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


def test_masks_become_thinned_polygons():
    outline = FakeTensor([[float(i), float(i % 7)] for i in range(200)])
    boxes = SimpleNamespace(
        xyxy=FakeTensor([[0.0, 0.0, 10.0, 10.0]]), conf=FakeTensor([0.9]), cls=FakeTensor([0.0]), id=None
    )
    result = SimpleNamespace(boxes=boxes, names={0: "robot"}, masks=SimpleNamespace(xy=[outline]))
    det = result_to_detections(result, 0, 0.0)[0]
    assert det.polygon is not None and 3 <= len(det.polygon) <= 48
    bad = SimpleNamespace(
        boxes=boxes, names={0: "robot"}, masks=SimpleNamespace(xy=[FakeTensor([[1.0, 2.0]])])
    )
    assert result_to_detections(bad, 0, 0.0)[0].polygon is None
