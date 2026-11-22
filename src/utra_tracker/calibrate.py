"""Click the four arena corners once so pixels map to maze feet."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from utra_tracker.maze import Homography, Maze, MazeError, Point, maze_corners, pixel_to_maze

CORNER_NAMES = (
    "loading zone corner",
    "far corner along the 8 ft side",
    "corner diagonal from the loading zone",
    "last corner on the loading zone 4 ft side",
)
WINDOW = "utra-track calibrate"


def save_calibration(path: Path, corners_px: Sequence[Point], image_size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "corners_px": [[round(x, 1), round(y, 1)] for x, y in corners_px],
        "image_size": list(image_size),
        "order": list(CORNER_NAMES),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: Path, image_size: tuple[int, int]) -> list[Point] | None:
    """Corner pixels scaled to image_size (width, height), or None when there is no file."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        corners = [(float(x), float(y)) for x, y in data["corners_px"]]
        width, height = data["image_size"]
    except (ValueError, KeyError, TypeError) as exc:
        raise MazeError(f"Calibration file {path} is invalid, rerun utra-track calibrate") from exc
    if len(corners) != 4:
        raise MazeError(f"Calibration file {path} needs 4 corners")
    sx, sy = image_size[0] / width, image_size[1] / height
    return [(x * sx, y * sy) for x, y in corners]


def arena_mapping(maze: Maze, calibration: Path, image_size: tuple[int, int]) -> tuple[Homography, bool]:
    """Pixel to maze homography and whether it came from a real calibration."""
    corners = load_calibration(calibration, image_size)
    if corners is None:
        w, h = image_size
        corners = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
        return pixel_to_maze(corners, maze), False
    return pixel_to_maze(corners, maze), True


def pick_corners(frame: Any, maze: Maze) -> list[Point] | None:
    """Show the frame and let the user click the 4 corners. Returns None if cancelled."""
    import cv2

    clicks: list[Point] = []

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    scale = min(1.0, 1400 / frame.shape[1], 850 / frame.shape[0])
    base = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            img = base.copy()
            for i, (x, y) in enumerate(clicks):
                p = (round(x * scale), round(y * scale))
                cv2.circle(img, p, 6, (0, 0, 255), -1)
                cv2.putText(
                    img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            if len(clicks) == 4:
                try:
                    to_px = pixel_to_maze(clicks, maze).inverse()
                except MazeError:
                    to_px = None
                if to_px is not None:
                    for a, b in maze.wall_segments():
                        pa, pb = to_px.apply(a), to_px.apply(b)
                        cv2.line(
                            img,
                            (round(pa[0] * scale), round(pa[1] * scale)),
                            (round(pb[0] * scale), round(pb[1] * scale)),
                            (0, 140, 255),
                            2,
                        )
                prompt = "Check the orange walls line up. Enter saves, right click undoes, Esc cancels"
            else:
                prompt = f"Click corner {len(clicks) + 1}: {CORNER_NAMES[len(clicks)]}  (right click undo)"
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (30, 30, 30), -1)
            cv2.putText(img, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key in (13, 10) and len(clicks) == 4:
                return list(clicks)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(WINDOW)


def grab_frame(source_value: Any) -> Any:
    """Read one frame from an image file, video, camera or stream."""
    import cv2

    from utra_tracker.sources import SourceError, open_capture, parse_source

    source = parse_source(source_value)
    if source.kind == "file" and Path(str(source.value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        frame = cv2.imread(str(source.value))
        if frame is None:
            raise SourceError(f"Could not read image {source.value}")
        return frame
    cap = open_capture(source)
    try:
        frame = None
        # Cameras often return dark frames while auto exposure settles.
        for _ in range(15 if source.is_live else 1):
            ok, latest = cap.read()
            if ok and latest is not None:
                frame = latest
        if frame is None:
            raise SourceError(f"Could not read a frame from {source.value}")
        return frame
    finally:
        cap.release()


def run_calibration(source_value: Any, maze: Maze, out: Path) -> bool:
    frame = grab_frame(source_value)
    corners = pick_corners(frame, maze)
    if corners is None:
        print("Calibration cancelled, nothing saved.")
        return False
    save_calibration(out, corners, (frame.shape[1], frame.shape[0]))
    print(f"Saved {out}. Corner order: {', '.join(CORNER_NAMES)}. Maze corners in ft: {maze_corners(maze)}")
    return True
