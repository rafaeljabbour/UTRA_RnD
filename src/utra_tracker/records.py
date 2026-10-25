"""Detection rows, the CSV writer and the run summary file."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CSV_COLUMNS = (
    "frame",
    "time_s",
    "track_id",
    "class_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
)
UNTRACKED_ID = -1


@dataclass(frozen=True)
class Detection:
    frame: int
    time_s: float
    track_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def as_row(self) -> list[Any]:
        coords = [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2, self.cx, self.cy)]
        return [
            self.frame,
            round(self.time_s, 3),
            self.track_id,
            self.class_name,
            round(self.conf, 4),
            *coords,
        ]


def result_to_detections(result: Any, frame: int, time_s: float) -> list[Detection]:
    """Convert one Ultralytics result into Detection rows."""
    boxes = result.boxes
    if boxes is None:
        return []
    xyxy = _as_list(boxes.xyxy)
    confs = _as_list(boxes.conf)
    classes = _as_list(boxes.cls)
    ids = _as_list(boxes.id) if boxes.id is not None else [UNTRACKED_ID] * len(xyxy)
    return [
        Detection(
            frame=frame,
            time_s=time_s,
            track_id=int(tid),
            class_name=str(result.names[int(cls)]),
            conf=float(conf),
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
        )
        for box, conf, cls, tid in zip(xyxy, confs, classes, ids, strict=True)
    ]


class CsvWriter:
    """Writes detections and flushes after every frame so a crash loses little."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)

    def write(self, detections: list[Detection]) -> None:
        self._writer.writerows(d.as_row() for d in detections)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> CsvWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_run_json(path: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    path.write_text(json.dumps({"config": config, "summary": summary}, indent=2), encoding="utf-8")


def _as_list(values: Any) -> list[Any]:
    return values.tolist() if hasattr(values, "tolist") else list(values)
