import csv
import json
from types import SimpleNamespace

from utra_tracker.records import (
    CSV_COLUMNS,
    UNTRACKED_ID,
    CsvWriter,
    Detection,
    result_to_detections,
    write_run_json,
)


class FakeTensor:
    """Mimics the tolist method of a torch tensor."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


def fake_result(ids):
    boxes = SimpleNamespace(
        xyxy=FakeTensor([[10.0, 20.0, 30.0, 60.0], [100.0, 100.0, 140.0, 120.0]]),
        conf=FakeTensor([0.91234, 0.5]),
        cls=FakeTensor([0.0, 1.0]),
        id=None if ids is None else FakeTensor(ids),
    )
    return SimpleNamespace(boxes=boxes, names={0: "white disk robot", 1: "robot car"})


def test_result_to_detections_with_ids():
    dets = result_to_detections(fake_result([3.0, 7.0]), frame=5, time_s=0.25)
    assert [d.track_id for d in dets] == [3, 7]
    first = dets[0]
    assert first.class_name == "white disk robot"
    assert first.frame == 5
    assert (first.cx, first.cy) == (20.0, 40.0)


def test_result_without_ids_marks_untracked():
    dets = result_to_detections(fake_result(None), frame=0, time_s=0.0)
    assert {d.track_id for d in dets} == {UNTRACKED_ID}


def test_result_without_boxes():
    assert result_to_detections(SimpleNamespace(boxes=None, names={}), 0, 0.0) == []


def test_row_order_matches_columns():
    det = Detection(1, 0.0333333, 2, "robot car", 0.876543, 1.04, 2.0, 3.0, 4.0)
    row = dict(zip(CSV_COLUMNS, det.as_row(), strict=True))
    assert row == {
        "frame": 1,
        "time_s": 0.033,
        "track_id": 2,
        "class_name": "robot car",
        "conf": 0.8765,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "cx": 2.0,
        "cy": 3.0,
    }


def test_csv_writer_roundtrip(tmp_path):
    path = tmp_path / "detections.csv"
    dets = result_to_detections(fake_result([1.0, 2.0]), frame=0, time_s=0.0)
    with CsvWriter(path) as writer:
        writer.write(dets)
        writer.write([])
        assert len(path.read_text().splitlines()) == 3
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert [r["track_id"] for r in rows] == ["1", "2"]


def test_csv_writer_close_twice(tmp_path):
    writer = CsvWriter(tmp_path / "d.csv")
    writer.close()
    writer.close()


def test_run_json(tmp_path):
    path = tmp_path / "run.json"
    write_run_json(path, {"source": 0}, {"frames": 10, "unique_tracks": 1})
    data = json.loads(path.read_text())
    assert data["config"]["source"] == 0
    assert data["summary"]["frames"] == 10
