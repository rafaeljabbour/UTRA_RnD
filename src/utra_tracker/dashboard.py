"""Autograder dashboard: the camera view with the maze drawn on it, next to a panel with the run."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from utra_tracker.maze import BLOCK, Cell, Homography, Maze, Point
from utra_tracker.referee import FINISHED, PASS, RUNNING, WAITING, Observation, Referee

PANEL_W = 440
VIEW_H = 720
MAP_PX = 52  # pixels per foot on the top down map

# BGR colours
BG = (30, 30, 30)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
WALL = (0, 140, 255)
START_ZONE = (220, 160, 60)
END_ZONE = (70, 200, 70)
PATH = (0, 215, 255)
ROBOT = (90, 230, 90)
CONTACT = (50, 50, 240)
START = (255, 120, 255)
GOOD = (90, 220, 90)
BAD = (60, 60, 235)

KEY_HELP = ("c wall contact tracking on / off", "r reset    h hide keys    q quit")
QUIT = "quit"
HELP_TOGGLE = "help"


def handle_key(key: int, referee: Referee) -> str | None:
    """Apply an operator hotkey. Returns a status message, QUIT or HELP_TOGGLE."""
    ch = chr(key).lower() if 0 <= key < 256 else ""
    if ch in ("q", "\x1b"):
        return QUIT
    if ch == "h":
        return HELP_TOGGLE
    if ch == "r":
        referee.reset()
        return "Run reset, place the robot in the start zone"
    if ch == "c":
        return f"Wall contact tracking {'ON' if referee.toggle_collisions() else 'OFF'}"
    return None


class Dashboard:
    def __init__(
        self, maze: Maze, to_maze: Homography, calibrated: bool, referee: Referee, title: str
    ) -> None:
        self.maze = maze
        self.to_px = to_maze.inverse()
        self.calibrated = calibrated
        self.start_cells = referee.start_cells
        self.end_cells = referee.end_cells
        self.title = title
        self.show_help = True
        self._message = ""
        self._message_until = 0.0
        self._walls = maze.wall_segments()

    def flash(self, message: str, seconds: float = 3.0) -> None:
        self._message = message
        self._message_until = time.monotonic() + seconds

    def render(self, frame: Any, referee: Referee, obs: Observation | None, fps: float) -> Any:
        scale = VIEW_H / frame.shape[0]
        view = cv2.resize(frame, (round(frame.shape[1] * scale), VIEW_H), interpolation=cv2.INTER_AREA)
        self._draw_camera(view, scale, referee, obs)
        return np.hstack([view, self._panel(referee, obs, fps)])

    def draw_map(self, referee: Referee, obs: Observation | None = None) -> Any:
        """Top down maze with zones, path, start, end and contacts. Also saved as path.png."""
        pad = 14
        w = round(self.maze.width_ft * MAP_PX) + 2 * pad
        h = round(self.maze.height_ft * MAP_PX) + 2 * pad
        img = np.full((h, w, 3), 245, np.uint8)

        def to(p: Point) -> tuple[int, int]:
            return (round(pad + p[0] * MAP_PX), round(pad + p[1] * MAP_PX))

        for c in range(self.maze.cols):
            for r in range(self.maze.rows):
                x0, y0, x1, y1 = self.maze.cell_rect((c, r))
                fill = None
                if self.maze.kind((c, r)) == BLOCK:
                    fill = (150, 150, 150)
                elif (c, r) in self.start_cells:
                    fill = (245, 220, 180)
                elif (c, r) in self.end_cells:
                    fill = (190, 240, 190)
                if fill:
                    cv2.rectangle(img, to((x0, y0)), to((x1, y1)), fill, -1)
                cv2.rectangle(img, to((x0, y0)), to((x1, y1)), (215, 215, 215), 1)
        self._zone_labels(img, to)
        for a, b in self._walls:
            cv2.line(img, to(a), to(b), WALL, 3)
        pts = [to((x, y)) for _, x, y in referee.path]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 160, 230), 2, cv2.LINE_AA)
        if referee.start_pos:
            cv2.circle(img, to(referee.start_pos), 7, START, -1)
            _text(img, "A", _offset(to(referee.start_pos), 8, -8), 0.5, (120, 0, 120), 2)
        for e in referee.events:
            colour = (0, 0, 200) if e.severity == "major" else (0, 120, 255)
            cv2.drawMarker(img, to((e.x_ft, e.y_ft)), colour, cv2.MARKER_TILTED_CROSS, 12, 2)
        if obs is not None and referee.state != FINISHED:
            poly = np.array([to(p) for p in obs.footprint], np.int32)
            cv2.polylines(img, [poly], True, CONTACT if referee.in_contact else (0, 150, 0), 2)
        if referee.state == FINISHED and referee.end_pos:
            cv2.circle(img, to(referee.end_pos), 7, (0, 0, 0), 2)
            _text(img, "end", _offset(to(referee.end_pos), 8, -8), 0.45, (0, 0, 0), 1)
        return img

    def _zone_labels(self, img: Any, to: Any) -> None:
        for cells, label, colour in (
            (self.start_cells, "start", (170, 100, 20)),
            (self.end_cells, "end", (30, 130, 30)),
        ):
            for cell in cells:
                x0, y0, _, _ = self.maze.cell_rect(cell)
                _text(img, label, to((x0 + 0.06, y0 + 0.28)), 0.38, colour, 1)

    def _draw_camera(self, view: Any, scale: float, referee: Referee, obs: Observation | None) -> None:
        def px(p: Point) -> tuple[int, int]:
            x, y = self.to_px.apply(p)
            return (round(x * scale), round(y * scale))

        def outline(cell: Cell, colour: tuple[int, int, int]) -> None:
            x0, y0, x1, y1 = self.maze.cell_rect(cell)
            _poly(view, [px(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], colour, 2)

        for cell in self.start_cells:
            outline(cell, START_ZONE)
        for cell in self.end_cells:
            outline(cell, END_ZONE)
        for a, b in self._walls:
            cv2.line(view, px(a), px(b), WALL, 2, cv2.LINE_AA)
        pts = [px((x, y)) for _, x, y in referee.path[-2000:]]
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], False, PATH, 2, cv2.LINE_AA)
        if referee.start_pos:
            cv2.circle(view, px(referee.start_pos), 8, START, -1)
            _text(view, "A", _offset(px(referee.start_pos), 10, -10), 0.7, START, 2)
        if obs is not None:
            colour = CONTACT if referee.in_contact else ROBOT
            _poly(view, [px(p) for p in obs.footprint], colour, 2)
            cv2.circle(view, px(obs.center), 4, colour, -1)
        if not self.calibrated:
            _banner(view, "NOT CALIBRATED: whole frame used as the maze. Run utra-track calibrate", BAD)
        elif referee.in_contact:
            _banner(view, f"WALL CONTACT  {referee.monitor.depth * 12:.1f} in", CONTACT)

    def _panel(self, referee: Referee, obs: Observation | None, fps: float) -> Any:
        panel = np.full((VIEW_H, PANEL_W, 3), BG, np.uint8)
        x, y = 14, 30
        _text(panel, self.title, (x, y), 0.6, TEXT, 2)
        y += 32
        label = {WAITING: "WAITING for the robot to move", RUNNING: "RUNNING", FINISHED: "FINISHED"}[
            referee.state
        ]
        _text(panel, label, (x, y), 0.6, {WAITING: DIM, RUNNING: GOOD, FINISHED: PATH}[referee.state], 2)
        limit = "" if referee.time_limit_s is None else f" / {_mmss(referee.time_limit_s)}"
        _text(panel, f"{_mmss(referee.elapsed_s)}{limit}", (PANEL_W - 130, y), 0.6, TEXT, 2)
        y += 26
        coll = "ON" if referee.collisions_on else "OFF"
        _text(panel, f"Wall contact tracking: {coll}", (x, y), 0.5, GOOD if referee.collisions_on else DIM, 1)
        _text(panel, f"{fps:4.1f} fps", (PANEL_W - 80, y), 0.45, DIM, 1)
        y += 12

        live = self.draw_map(referee, obs)
        map_w = PANEL_W - 2 * x
        map_h = round(live.shape[0] * map_w / live.shape[1])
        panel[y : y + map_h, x : x + map_w] = cv2.resize(live, (map_w, map_h), interpolation=cv2.INTER_AREA)
        y += map_h + 28

        m = referee.metrics()
        started = "--" if referee.state == WAITING else "yes" if m["started_in_start_zone"] else "NO"
        reached = "--" if m["reached_end_zone_s"] is None else f"at {m['reached_end_zone_s']:.1f} s"
        rows = [
            (f"Started in start zone: {started}", BAD if started == "NO" else TEXT),
            (f"In end zone now: {m['end_zone_share']:.0%}   reached {reached}", TEXT),
            (f"Distance {m['distance_ft']:.1f} ft   1 ft steps {m['steps']}", TEXT),
        ]
        if referee.collisions_on or referee.events:
            rows.append(
                (f"Wall contacts  minor {m['minor_collisions']}  major {m['major_collisions']}", TEXT)
            )
        for row, colour in rows:
            _text(panel, row, (x, y), 0.5, colour, 1)
            y += 24

        if referee.state == FINISHED:
            outcome = referee.outcome
            colour = GOOD if outcome == PASS else BAD
            y += 14
            _text(panel, outcome, (x, y + 30), 1.6, colour, 4)
            _text(panel, referee.finish_reason or "", (x + 150, y + 22), 0.5, TEXT, 1)
            y += 60
        if self._message and time.monotonic() < self._message_until:
            _text(panel, self._message, (x, y + 8), 0.5, PATH, 2)

        lines = KEY_HELP if self.show_help else ("h show keys",)
        y = VIEW_H - 10 - 19 * (len(lines) - 1)
        for line in lines:
            _text(panel, line, (x, y), 0.45, DIM, 1)
            y += 19
        return panel


def _poly(img: Any, pts: list[tuple[int, int]], colour: tuple[int, int, int], thick: int) -> None:
    cv2.polylines(img, [np.array(pts, np.int32)], True, colour, thick, cv2.LINE_AA)


def _offset(p: tuple[int, int], dx: int, dy: int) -> tuple[int, int]:
    return (p[0] + dx, p[1] + dy)


def _text(img: Any, text: str, org: tuple[int, int], size: float, colour: Any, thick: int) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def _banner(img: Any, text: str, colour: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), colour, -1)
    _text(img, text, (10, 24), 0.6, (255, 255, 255), 2)


def _mmss(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"
