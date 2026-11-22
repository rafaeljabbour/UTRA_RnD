"""Grading dashboard: the camera view with the maze drawn on it, next to a panel with the run and rubric."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from utra_tracker.maze import BLOCK, Homography, Maze, Point
from utra_tracker.referee import FINISHED, RUNNING, WAITING, Observation, Referee
from utra_tracker.rubric import items_for, suggest

PANEL_W = 470
VIEW_H = 720
MAP_PX = 52  # pixels per foot on the top down map

# BGR colours
BG = (30, 30, 30)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
WALL = (0, 140, 255)
LOAD = (200, 150, 50)
DROP = (70, 190, 70)
TARGET = (0, 255, 120)
PATH = (0, 215, 255)
ROBOT = (90, 230, 90)
CONTACT = (50, 50, 240)
START = (255, 120, 255)
GOOD = (90, 220, 90)
BAD = (60, 60, 235)

KEY_HELP = (
    "space start / stop    r reset    q finish and save",
    "c collision tracking on / off    b cycle target B",
    "a log assistance    m / n judge minor / major hit",
    "u undo last hit or assist    h hide these keys",
)
QUIT = "quit"
HELP_TOGGLE = "help"


def handle_key(key: int, referee: Referee, t: float) -> str | None:
    """Apply a dashboard hotkey to the referee. Returns a status message or QUIT."""
    ch = chr(key).lower() if 0 <= key < 256 else ""
    if ch in ("q", "\x1b"):
        return QUIT
    if ch == " ":
        if referee.state == WAITING:
            referee.start(t)
            return "Run started"
        if referee.state == RUNNING:
            referee.finish(t)
            return "Run finished, press q to save or r to reset"
        return "Run already finished, press r to reset"
    if ch == "h":
        return HELP_TOGGLE
    if ch == "r":
        referee.reset()
        return "Run reset"
    if ch == "c":
        return f"Collision tracking {'ON' if referee.toggle_collisions() else 'OFF'}"
    if ch == "a":
        referee.add_assist()
        return f"Assistance logged ({len(referee.assists)})"
    if ch in ("m", "n"):
        severity = "minor" if ch == "m" else "major"
        referee.add_judge_collision(severity)
        return f"Judge logged a {severity} collision"
    if ch == "u":
        undone = referee.undo_last()
        return f"Removed last {undone}" if undone else "Nothing to undo"
    if ch == "b":
        return f"Target {referee.cycle_target() or 'cleared'}"
    return None


class Dashboard:
    def __init__(
        self, maze: Maze, to_maze: Homography, calibrated: bool, milestone: int, team: str, run: str
    ) -> None:
        self.maze = maze
        self.to_px = to_maze.inverse()
        self.calibrated = calibrated
        self.milestone = milestone
        self.title = f"MIE444 M{milestone}   team {team or '?'}   run {run or '?'}"
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
        panel = self._panel(referee, obs, fps)
        return np.hstack([view, panel])

    def draw_map(self, referee: Referee, obs: Observation | None = None) -> Any:
        """Top down maze with the path, start, end and contacts. Also saved as path.png."""
        pad = 14
        w = round(self.maze.width_ft * MAP_PX) + 2 * pad
        h = round(self.maze.height_ft * MAP_PX) + 2 * pad
        img = np.full((h, w, 3), 245, np.uint8)

        def to(p: Point) -> tuple[int, int]:
            return (round(pad + p[0] * MAP_PX), round(pad + p[1] * MAP_PX))

        for c in range(self.maze.cols):
            for r in range(self.maze.rows):
                x0, y0, x1, y1 = self.maze.cell_rect((c, r))
                kind = self.maze.kind((c, r))
                if kind == BLOCK:
                    cv2.rectangle(img, to((x0, y0)), to((x1, y1)), (150, 150, 150), -1)
                elif kind == "L":
                    cv2.rectangle(img, to((x0, y0)), to((x1, y1)), (240, 215, 170), -1)
                cv2.rectangle(img, to((x0, y0)), to((x1, y1)), (215, 215, 215), 1)
        for i, cell in enumerate(self.maze.drop_cells):
            x0, y0, x1, y1 = self.maze.cell_rect(cell)
            target = referee.target_b == i
            colour = TARGET if target else DROP
            cv2.rectangle(img, to((x0, y0)), to((x1, y1)), colour, 3 if target else 1)
            _text(img, f"B{i + 1}", to((x0 + 0.08, y0 + 0.3)), 0.45, colour, 1)
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
        end = referee.end_pos if referee.state == FINISHED else None
        if end:
            cv2.circle(img, to(end), 7, (0, 0, 0), 2)
            _text(img, "end", _offset(to(end), 8, -8), 0.45, (0, 0, 0), 1)
        return img

    def _draw_camera(self, view: Any, scale: float, referee: Referee, obs: Observation | None) -> None:
        def px(p: Point) -> tuple[int, int]:
            x, y = self.to_px.apply(p)
            return (round(x * scale), round(y * scale))

        for cell in self.maze.loading_cells:
            _poly(view, [px(c) for c in _rect_corners(self.maze.cell_rect(cell))], LOAD, 1)
        for i, cell in enumerate(self.maze.drop_cells):
            target = referee.target_b == i
            colour = TARGET if target else DROP
            corners = [px(c) for c in _rect_corners(self.maze.cell_rect(cell))]
            _poly(view, corners, colour, 3 if target else 1)
            _text(view, f"B{i + 1}", _offset(corners[0], 4, 18), 0.55, colour, 2)
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
        state_colour = {WAITING: DIM, RUNNING: GOOD, FINISHED: PATH}[referee.state]
        label = {
            WAITING: "WAITING  (move or space)",
            RUNNING: "RUNNING",
            FINISHED: "FINISHED  q saves, r resets",
        }[referee.state]
        _text(panel, label, (x, y), 0.6, state_colour, 2)
        clock = f"{_mmss(referee.elapsed_s)} / {_mmss(referee.time_limit_s)}"
        _text(panel, clock, (PANEL_W - 150, y), 0.65, BAD if referee.over_time else TEXT, 2)
        y += 28
        coll = "ON" if referee.collisions_on else "OFF (judge tally with m / n)"
        _text(panel, f"Collision tracking: {coll}", (x, y), 0.5, GOOD if referee.collisions_on else PATH, 1)
        _text(panel, f"{fps:4.1f} fps", (PANEL_W - 80, y), 0.45, DIM, 1)
        y += 12

        live = self.draw_map(referee, obs)
        map_w = PANEL_W - 2 * x
        map_h = round(live.shape[0] * map_w / live.shape[1])
        panel[y : y + map_h, x : x + map_w] = cv2.resize(live, (map_w, map_h), interpolation=cv2.INTER_AREA)
        y += map_h + 24

        m = referee.metrics()
        lz = "--" if m["lz_reached_s"] is None else f"{m['lz_reached_s']:.1f} s"
        rows = [
            f"Distance {m['distance_ft']:.1f} ft   1 ft steps {m['steps']}   turns {m['turns']}",
            f"Speed {m['speed_steps_per_min']:.1f} steps/min   cells visited {m['unique_cells']}",
            f"Collisions  minor {m['minor_collisions']}  major {m['major_collisions']}"
            f"   assists {m['assists']}",
            f"Loading zone reached {lz}   target {m['target_b'] or 'not set (b)'}",
        ]
        for row in rows:
            _text(panel, row, (x, y), 0.48, TEXT, 1)
            y += 22
        y += 6
        _text(panel, "Suggested marks (judge confirms after the run)", (x, y), 0.48, PATH, 1)
        y += 22
        labels = {i.key: i for i in items_for(self.milestone)}
        for key, (score, _note) in suggest(self.milestone, m).items():
            item = labels[key]
            if not item.max:
                continue
            shown = "rank" if score is None else f"{score:g}"
            _text(panel, f"{item.label[:44]:<44}", (x, y), 0.45, TEXT, 1)
            _text(panel, f"{shown}/{item.max:g}", (PANEL_W - 70, y), 0.45, TEXT, 1)
            y += 20

        if self._message and time.monotonic() < self._message_until:
            _text(panel, self._message, (x, y + 8), 0.52, PATH, 2)
        y += 30
        lines = KEY_HELP if self.show_help else ("h show hotkeys",)
        y = max(y, VIEW_H - 8 - 19 * (len(lines) - 1))
        for line in lines:
            _text(panel, line, (x, y), 0.45, DIM, 1)
            y += 19
        return panel


def _rect_corners(rect: tuple[float, float, float, float]) -> list[Point]:
    x0, y0, x1, y1 = rect
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


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
