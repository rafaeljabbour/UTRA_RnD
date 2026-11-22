from types import SimpleNamespace

import pytest

from tests_support import finished_metrics
from utra_tracker.calibrate import arena_mapping, load_calibration, save_calibration
from utra_tracker.cli import _mark, main
from utra_tracker.maze import Maze, MazeError
from utra_tracker.records import Detection, result_to_detections
from utra_tracker.rubric import build_sheet, load_sheet, save_sheet
from utra_tracker.session import detection_to_observation


def scripted(answers):
    answers = iter(answers)
    return lambda prompt: next(answers)


def write_sheet(tmp_path, milestone=3):
    path = tmp_path / "scoresheet.yaml"
    save_sheet(path, build_sheet(milestone, "7", "1", finished_metrics()))
    return path


def test_mark_accepts_suggestions_and_judge_values(tmp_path, capsys):
    path = write_sheet(tmp_path)
    # judge, team, run, control, then the 15 items of milestone 3 in order
    items = ["", "3", "", "4", "", "", "4", "3", "2", "4", "2", "2", "", "", "1"]
    assert _mark(path, scripted(["TA Kim", "", "2", "autonomous", *items])) == 0
    sheet = load_sheet(path)
    assert sheet["judge"] == "TA Kim" and sheet["run"] == "2" and sheet["control"] == "autonomous"
    assert sheet["items"]["functional"]["judge"] == 8  # Enter confirmed the suggestion
    assert sheet["items"]["build_quality"]["judge"] == 3
    assert sheet["items"]["speed"]["judge"] is None  # waits for the leaderboard
    assert sheet["totals"]["pending"] == ["speed"]
    assert "Total" in capsys.readouterr().out


def test_mark_rejects_out_of_range_then_accepts(tmp_path, capsys):
    path = write_sheet(tmp_path, milestone=2)
    items = ["", "9", "abc", "1", "", "", "", "", "", "", "-1"]
    assert _mark(path, scripted(["", "", "", *items])) == 0
    sheet = load_sheet(path)
    assert sheet["items"]["build_quality"]["judge"] == 1
    assert sheet["items"]["usb_deduction"]["judge"] == -1
    out = capsys.readouterr().out
    assert "between 0 and 2" in out and "enter a number" in out


def test_grade_show_and_missing_sheet(tmp_path, capsys):
    write_sheet(tmp_path)
    assert main(["grade", str(tmp_path), "--show"]) == 0
    assert "MIE444 Milestone 3" in capsys.readouterr().out
    assert main(["grade", str(tmp_path / "nope"), "--show"]) == 2


def test_leaderboard_writes_speed(tmp_path, capsys):
    for i, speed in enumerate([12.0, 6.0]):
        run = tmp_path / f"run{i}"
        run.mkdir()
        save_sheet(
            run / "scoresheet.yaml", build_sheet(3, str(i), "1", finished_metrics(speed_steps_per_min=speed))
        )
    assert main(["leaderboard", "--runs", str(tmp_path), "--milestone", "3"]) == 0
    assert load_sheet(tmp_path / "run0")["items"]["speed"]["auto"] == 5
    assert load_sheet(tmp_path / "run1")["items"]["speed"]["auto"] == 0
    assert main(["leaderboard", "--runs", str(tmp_path), "--milestone", "2"]) == 1


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
