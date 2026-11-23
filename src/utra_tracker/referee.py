"""Autograder for one run: did the robot start in the start zone and end in the end zone.

Plain Python, no numpy or OpenCV, so it is unit tested in CI. Positions are maze feet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from utra_tracker.config import GradingConfig
from utra_tracker.maze import Cell, Maze, Point

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"
PASS, FAIL = "PASS", "FAIL"
REACHED_END, TIME_LIMIT, STOPPED = "settled in end zone", "time limit", "stopped before reaching the end zone"

SMOOTHING = 0.5  # weight of the newest centre in the moving average
JITTER_FT = 0.05  # moves smaller than this are detector noise, not distance
STEP_INSET_FT = 0.1  # a 1 ft step counts once the centre is this far inside the new cell
SETTLE_FT = 0.15  # the robot counts as still while it stays within this distance
ARENA_MARGIN_FT = 0.5  # detections further outside the arena are ignored


@dataclass(frozen=True)
class Observation:
    """One robot detection in maze feet."""

    center: Point
    footprint: tuple[Point, ...]
    conf: float = 1.0


@dataclass
class ContactEvent:
    start_s: float
    end_s: float
    depth_ft: float
    x_ft: float
    y_ft: float
    severity: str  # minor or major

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


class ContactMonitor:
    """Debounces wall contact into events and grades each one minor or major."""

    def __init__(self, g: GradingConfig) -> None:
        self.margin = g.contact_margin_ft
        self.major_depth = g.major_depth_ft
        self.major_duration = g.major_duration_s
        self.on_frames = g.contact_frames
        self.off_frames = g.release_frames
        self._hits = 0
        self._misses = 0
        self._pending: dict | None = None
        self._active: dict | None = None
        self.depth = 0.0

    @property
    def in_contact(self) -> bool:
        return self._active is not None

    def update(self, t: float, depth: float | None, pos: Point | None) -> ContactEvent | None:
        """Feed one frame. depth None means the robot was not seen. Returns a finished event."""
        self.depth = depth or 0.0
        if depth is not None and depth > self.margin and pos is not None:
            self._misses = 0
            target = self._active or self._pending
            if target is None:
                self._pending = target = {"start": t, "depth": depth, "pos": pos, "last": t}
            elif depth > target["depth"]:
                target.update(depth=depth, pos=pos)
            target["last"] = t
            if self._active is None:
                self._hits += 1
                if self._hits >= self.on_frames:
                    self._active, self._pending = self._pending, None
            return None
        self._hits = 0
        self._pending = None
        if self._active is not None:
            self._misses += 1
            if self._misses >= self.off_frames:
                return self.close()
        return None

    def close(self) -> ContactEvent | None:
        """End the contact in progress, if any."""
        active, self._active, self._pending = self._active, None, None
        self._hits = self._misses = 0
        if active is None:
            return None
        duration = active["last"] - active["start"]
        major = active["depth"] >= self.major_depth or duration >= self.major_duration
        return ContactEvent(
            start_s=active["start"],
            end_s=active["last"],
            depth_ft=active["depth"],
            x_ft=active["pos"][0],
            y_ft=active["pos"][1],
            severity="major" if major else "minor",
        )


class Referee:
    """One autograded run: waiting, running, finished. Starts and ends on its own."""

    def __init__(self, maze: Maze, grading: GradingConfig) -> None:
        self.maze = maze
        self.g = grading
        self.start_cells = maze.zone_cells(grading.start_zone)
        self.end_cells = maze.zone_cells(grading.end_zone)
        self.collisions_on = grading.collisions
        self.time_limit_s = grading.time_limit_s
        self.reset()

    def reset(self) -> None:
        self.state = WAITING
        self.finish_reason: str | None = None
        self.now = 0.0
        self.start_s: float | None = None
        self.end_s: float | None = None
        self.start_pos: Point | None = None
        self.start_share = 0.0
        self.end_pos: Point | None = None
        self.end_share = 0.0
        self.end_share_now = 0.0
        self.reached_end_s: float | None = None
        self.position: Point | None = None
        self.last_obs: Observation | None = None
        self._first: Observation | None = None
        self.path: list[tuple[float, float, float]] = []
        self.distance_ft = 0.0
        self.steps = 0
        self.cell: Cell | None = None
        self.cells_visited: set[Cell] = set()
        self._anchor: Point | None = None
        self._settle: tuple[Point, float] | None = None
        self.events: list[ContactEvent] = []
        self.monitor = ContactMonitor(self.g)
        self.collisions_toggled = False

    @property
    def elapsed_s(self) -> float:
        if self.start_s is None:
            return 0.0
        end = self.end_s if self.end_s is not None else self.now
        return max(end - self.start_s, 0.0)

    @property
    def in_contact(self) -> bool:
        return self.collisions_on and self.monitor.in_contact

    @property
    def outcome(self) -> str | None:
        if self.state != FINISHED:
            return None
        return PASS if self.started_in_zone and self.ended_in_zone else FAIL

    @property
    def started_in_zone(self) -> bool:
        return self.start_share >= self.g.zone_share

    @property
    def ended_in_zone(self) -> bool:
        return self.end_share >= self.g.zone_share

    def update(self, t: float, obs: Observation | None) -> ContactEvent | None:
        """Feed one frame. Returns a wall contact event when one just ended."""
        self.now = t
        if obs is not None:
            self.last_obs = obs
            self.position = obs.center if self.position is None else _mix(self.position, obs.center)
        if self.state == WAITING:
            if obs is not None and self._first is None:
                self._first = Observation(self.position, obs.footprint, obs.conf)
            if (
                self._first is not None
                and math.dist(self._first.center, self.position) >= self.g.start_move_ft
            ):
                self._start(t)
            return None
        if self.state != RUNNING:
            return None
        event = None
        if obs is not None:
            self._advance(t, self.position)
            self._check_end_zone(t, obs)
        if self.collisions_on:
            depth = self.maze.penetration(obs.footprint) if obs is not None else None
            event = self.monitor.update(t, depth, obs.center if obs is not None else None)
            if event is not None:
                self.events.append(event)
        if self.state == RUNNING and self.time_limit_s is not None and self.elapsed_s >= self.time_limit_s:
            self.finish(t, TIME_LIMIT)
        return event

    def finish(self, t: float | None = None, reason: str = STOPPED) -> None:
        """End the run. The end point is the last place the robot was seen."""
        if self.state != RUNNING:
            return
        self.now = self.now if t is None else t
        self.state = FINISHED
        self.finish_reason = reason
        self.end_s = self.now
        self.end_pos = self.position
        if self.last_obs is not None:
            self.end_share = self.maze.zone_coverage(self.last_obs.footprint, self.end_cells)
        event = self.monitor.close()
        if event is not None:
            self.events.append(event)

    def toggle_collisions(self) -> bool:
        self.collisions_on = not self.collisions_on
        self.collisions_toggled = True
        if not self.collisions_on:
            event = self.monitor.close()
            if event is not None:
                self.events.append(event)
        return self.collisions_on

    def metrics(self) -> dict:
        """JSON friendly result of the run."""
        end = self.end_pos or self.position
        minor = sum(e.severity == "minor" for e in self.events)
        return {
            "outcome": self.outcome,
            "state": self.state,
            "finish_reason": self.finish_reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "time_limit_s": self.time_limit_s,
            "started_in_start_zone": self.started_in_zone,
            "start_zone_share": round(self.start_share, 3),
            "start_ft": _round_point(self.start_pos),
            "start_cell": _cell_list(self.start_pos, self.maze),
            "ended_in_end_zone": self.ended_in_zone,
            "end_zone_share": round(self.end_share if self.state == FINISHED else self.end_share_now, 3),
            "reached_end_zone_s": None if self.reached_end_s is None else round(self.reached_end_s, 2),
            "end_ft": _round_point(end),
            "end_cell": _cell_list(end, self.maze),
            "distance_ft": round(self.distance_ft, 2),
            "steps": self.steps,
            "unique_cells": len(self.cells_visited),
            "collision_tracking": self.collisions_on,
            "collision_tracking_toggled": self.collisions_toggled,
            "minor_collisions": minor,
            "major_collisions": len(self.events) - minor,
            "collisions": [e.to_dict() for e in self.events],
        }

    def _start(self, t: float) -> None:
        self.state = RUNNING
        self.start_s = t
        self.start_pos = self._first.center
        self.start_share = self.maze.zone_coverage(self._first.footprint, self.start_cells)
        self._advance(t, self.start_pos)
        self._advance(t, self.position)

    def _check_end_zone(self, t: float, obs: Observation) -> None:
        self.end_share_now = self.maze.zone_coverage(obs.footprint, self.end_cells)
        if self.end_share_now < self.g.zone_share:
            self._settle = None
            return
        if self.reached_end_s is None:
            self.reached_end_s = t - self.start_s
        pos = self.position
        if self._settle is None or math.dist(self._settle[0], pos) > SETTLE_FT:
            self._settle = (pos, t)
        elif t - self._settle[1] >= self.g.settle_s:
            self.finish(t, REACHED_END)

    def _advance(self, t: float, p: Point) -> None:
        self.path.append((t, p[0], p[1]))
        if self._anchor is None:
            self._anchor = p
        step = math.dist(self._anchor, p)
        if step >= JITTER_FT:
            self.distance_ft += step
            self._anchor = p
        cell = self.maze.inset_cell(p, STEP_INSET_FT)
        if cell is not None:
            if self.cell is not None and cell != self.cell:
                # A lost track can skip cells, so count the grid distance, not one.
                self.steps += abs(cell[0] - self.cell[0]) + abs(cell[1] - self.cell[1])
            self.cell = cell
            self.cells_visited.add(cell)


def pick_robot(candidates: Sequence[Observation], maze: Maze, last: Point | None) -> Observation | None:
    """Choose the one robot to grade: confident, inside the arena and near where it was."""
    best, best_score = None, -math.inf
    for obs in candidates:
        x, y = obs.center
        m = ARENA_MARGIN_FT
        if not (-m <= x <= maze.width_ft + m and -m <= y <= maze.height_ft + m):
            continue
        score = obs.conf - (0.25 * math.dist(last, obs.center) if last is not None else 0.0)
        if score > best_score:
            best, best_score = obs, score
    return best


def _mix(a: Point, b: Point) -> Point:
    return (a[0] + (b[0] - a[0]) * SMOOTHING, a[1] + (b[1] - a[1]) * SMOOTHING)


def _round_point(p: Point | None) -> list[float] | None:
    return None if p is None else [round(p[0], 3), round(p[1], 3)]


def _cell_list(p: Point | None, maze: Maze) -> list[int] | None:
    cell = maze.cell_at(p) if p is not None else None
    return None if cell is None else [cell[0], cell[1]]
