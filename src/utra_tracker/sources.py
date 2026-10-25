"""Parse and open video sources: camera index, file path or stream URL."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


class SourceError(RuntimeError):
    """Raised when a source cannot be parsed or opened."""


@dataclass(frozen=True)
class Source:
    kind: Literal["camera", "file", "stream"]
    value: int | str

    @property
    def is_live(self) -> bool:
        return self.kind != "file"


def parse_source(raw: int | str) -> Source:
    """Classify a config or CLI value as camera, file or stream."""
    if isinstance(raw, bool):
        raise SourceError("Source must be a camera index, file path or URL")
    if isinstance(raw, int):
        if raw < 0:
            raise SourceError("Camera index must be >= 0")
        return Source("camera", raw)
    text = str(raw).strip()
    if not text:
        raise SourceError("Source is empty")
    if text.isdigit():
        return Source("camera", int(text))
    if text.lower().startswith(STREAM_SCHEMES):
        return Source("stream", text)
    return Source("file", text)


def open_capture(source: Source) -> Any:
    """Open a cv2.VideoCapture for the source or raise SourceError."""
    cv2 = _cv2()
    if source.kind == "camera":
        # DirectShow opens USB cameras in about a second on Windows, MSMF can take much longer.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source.value), backend)
    elif source.kind == "stream":
        cap = cv2.VideoCapture(str(source.value), cv2.CAP_FFMPEG)
    else:
        if not Path(str(source.value)).is_file():
            raise SourceError(f"Video file not found: {source.value}")
        cap = cv2.VideoCapture(str(source.value))
    if not cap.isOpened():
        cap.release()
        raise SourceError(f"Could not open {source.kind} source: {source.value}")
    return cap


def probe_cameras(max_index: int = 9) -> list[tuple[int, int, int]]:
    """Return index, width and height for each camera that delivers a frame."""
    found = []
    for index in range(max_index + 1):
        try:
            cap = open_capture(Source("camera", index))
        except SourceError:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            found.append((index, width, height))
    return found


def _cv2() -> Any:
    os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
    import cv2

    return cv2
