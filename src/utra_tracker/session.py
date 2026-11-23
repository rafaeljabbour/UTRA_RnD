"""Glue between the detector loop and the autograder referee, dashboard and result files."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from utra_tracker.config import Config
from utra_tracker.maze import Homography, Maze, Point, correct_parallax, polygon_centroid
from utra_tracker.records import Detection
from utra_tracker.referee import FINISHED, Observation, Referee, pick_robot

TRACK_COLUMNS = (
    "frame",
    "time_s",
    "state",
    "seen",
    "x_ft",
    "y_ft",
    "col",
    "row",
    "end_zone_share",
    "contact",
)
RESULT_HOLD_S = 3.0  # keep the PASS or FAIL on screen this long before closing
Parallax = tuple[Point, float, float]


def detection_to_observation(
    det: Detection, to_maze: Homography, footprint: str, parallax: Parallax | None = None
) -> Observation:
    """Map a pixel detection to maze feet, using the mask outline when there is one."""
    pixels = det.polygon if footprint == "mask" and det.polygon else det.box_corners
    points = [to_maze.apply(p) for p in pixels]
    if parallax is not None:
        nadir, camera_h, robot_h = parallax
        points = [correct_parallax(p, nadir, camera_h, robot_h) for p in points]
    return Observation(center=polygon_centroid(points), footprint=tuple(points), conf=det.conf)


class GradingSession:
    """Everything the autograder needs, built once the frame size is known."""

    def __init__(self, cfg: Config, frame_size: tuple[int, int], run_dir: Path) -> None:
        from utra_tracker.calibrate import arena_mapping
        from utra_tracker.dashboard import Dashboard

        g = cfg.grading
        self.cfg = cfg
        self.maze = Maze(cfg.arena.layout, cfg.arena.cell_ft)
        self.to_maze, self.calibrated = arena_mapping(self.maze, Path(cfg.arena.calibration), frame_size)
        if not self.calibrated:
            print(
                f"Warning: {cfg.arena.calibration} not found, so the whole frame is taken as the maze. "
                "Run 'utra-track calibrate' for accurate results."
            )
        self.parallax: Parallax | None = None
        if cfg.arena.camera_height_ft is not None:
            nadir = (self.maze.width_ft / 2, self.maze.height_ft / 2)
            self.parallax = (nadir, cfg.arena.camera_height_ft, cfg.arena.robot_height_ft)
        self.referee = Referee(self.maze, g)
        title = f"Autograder   team {g.team or '?'}   run {g.run or '?'}"
        self.dashboard = Dashboard(self.maze, self.to_maze, self.calibrated, self.referee, title)
        self.obs: Observation | None = None
        self._finished_at: float | None = None
        self._file = (run_dir / "track.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TRACK_COLUMNS)

    def update(self, frame: int, time_s: float, detections: list[Detection]) -> None:
        footprint = self.cfg.grading.footprint
        candidates = [detection_to_observation(d, self.to_maze, footprint, self.parallax) for d in detections]
        ref = self.referee
        self.obs = pick_robot(candidates, self.maze, ref.position)
        was_running = ref.state != FINISHED
        event = ref.update(time_s, self.obs)
        if event is not None:
            self.dashboard.flash(f"{event.severity.upper()} wall contact, {event.depth_ft * 12:.1f} in deep")
        if was_running and ref.state == FINISHED:
            self._finished_at = time.monotonic()
            print(f"Run finished: {ref.outcome}, {ref.finish_reason}")
        pos = ref.position
        cell = self.maze.cell_at(pos) if pos else None
        self._writer.writerow(
            [
                frame,
                round(time_s, 3),
                ref.state,
                int(self.obs is not None),
                "" if pos is None else round(pos[0], 3),
                "" if pos is None else round(pos[1], 3),
                "" if cell is None else cell[0],
                "" if cell is None else cell[1],
                round(ref.end_share_now, 3),
                int(ref.in_contact),
            ]
        )
        self._file.flush()

    def done(self, showing: bool) -> bool:
        """True once the run has finished and its result has been on screen long enough."""
        if self._finished_at is None:
            return False
        return not showing or time.monotonic() - self._finished_at >= RESULT_HOLD_S

    def render(self, frame: Any, fps: float) -> Any:
        return self.dashboard.render(frame, self.referee, self.obs, fps)

    def key(self, key: int) -> bool:
        """Handle a hotkey. Returns True when the operator wants to quit."""
        from utra_tracker.dashboard import HELP_TOGGLE, QUIT, handle_key

        message = handle_key(key, self.referee)
        if message == QUIT:
            return True
        if message == HELP_TOGGLE:
            self.dashboard.show_help = not self.dashboard.show_help
        elif message:
            if self.referee.state != FINISHED:
                self._finished_at = None
            self.dashboard.flash(message)
            print(message)
        return False

    def close(self, run_dir: Path, time_s: float) -> Path | None:
        """Finish the run and write result.json, results.csv and path.png. None if it never started."""
        import cv2

        from utra_tracker.result import build_result, format_result, save_result

        if not self._file.closed:
            self._file.close()
        ref = self.referee
        ref.finish(time_s)
        if ref.start_s is None:
            print("The robot never moved, so no result was written.")
            return None
        g = self.cfg.grading
        result = build_result(g.team, g.run, ref.metrics(), self.calibrated)
        path = save_result(run_dir, result)
        cv2.imwrite(str(run_dir / "path.png"), self.dashboard.draw_map(ref))
        print(format_result(result))
        return path
