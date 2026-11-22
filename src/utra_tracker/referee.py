"""Turn per frame robot positions into the run facts the rubric needs.

Plain Python, no numpy or OpenCV, so the whole referee is unit tested in CI.
All positions are maze feet from utra_tracker.maze.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from utra_tracker.config import GradingConfig
from utra_tracker.maze import Cell, Maze, Point

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"
TIME_LIMITS_S = {2: 300.0, 3: 480.0}

SMOOTHING = 0.5  # weight of the newest centre in the moving average
JITTER_FT = 0.05  # moves smaller than this are detector noise, not distance
STEP_INSET_FT = 0.1  # a 1 ft step counts once the centre is this far inside the new cell
TURN_SEGMENT_FT = 0.5  # heading is measured over path pieces this long
TURN_DEG = 45.0
DROVE_FT = 1.0
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
    source: str  # camera or judge

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("start_s", "end_s", "depth_ft", "x_ft", "y_ft"):
            data[key] = round(data[key], 3)
        return data


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
            source="camera",
        )


class Referee:
    """State machine for one graded run: waiting, running, finished."""

    def __init__(self, maze: Maze, grading: GradingConfig) -> None:
        self.maze = maze
        self.g = grading
        self.collisions_on = grading.collisions
        self.time_limit_s = grading.time_limit_s or TIME_LIMITS_S.get(grading.milestone, 480.0)
        self.target_b: int | None = None
        self.reset()

    def reset(self) -> None:
        self.state = WAITING
        self.now = 0.0
        self.start_s: float | None = None
        self.end_s: float | None = None
        self.start_pos: Point | None = None
        self.end_pos: Point | None = None
        self.end_footprint: tuple[Point, ...] | None = None
        self.position: Point | None = None
        self.last_obs: Observation | None = None
        self.first_seen: Point | None = None
        self.path: list[tuple[float, float, float]] = []
        self.distance_ft = 0.0
        self.steps = 0
        self.cell: Cell | None = None
        self.cells_visited: set[Cell] = set()
        self.turns = 0
        self.lz_reached_s: float | None = None
        self._outside_lz = False
        self._anchor: Point | None = None
        self._heading_anchor: Point | None = None
        self._heading: float | None = None
        self.events: list[ContactEvent] = []
        self.assists: list[float] = []
        self._log: list[tuple[str, object]] = []
        self.monitor = ContactMonitor(self.g)
        self.collisions_toggled = False

    @property
    def elapsed_s(self) -> float:
        if self.start_s is None:
            return 0.0
        end = self.end_s if self.end_s is not None else self.now
        return max(end - self.start_s, 0.0)

    @property
    def over_time(self) -> bool:
        return self.elapsed_s > self.time_limit_s

    @property
    def in_contact(self) -> bool:
        return self.collisions_on and self.monitor.in_contact

    def update(self, t: float, obs: Observation | None) -> ContactEvent | None:
        """Feed one frame. Returns a wall contact event when one just ended."""
        self.now = t
        if obs is not None:
            self.last_obs = obs
            if self.position is None:
                self.position = obs.center
            else:
                self.position = _mix(self.position, obs.center, SMOOTHING)
        if self.state == WAITING:
            if self.position is not None:
                if self.first_seen is None:
                    self.first_seen = self.position
                moved = math.dist(self.first_seen, self.position)
                if self.g.auto_start and moved >= self.g.start_move_ft:
                    self.start(t, origin=self.first_seen)
            return None
        if self.state != RUNNING:
            return None
        if obs is not None:
            if self.start_pos is None:
                self.start_pos = self.position
            self._advance(t, self.position)
        if not self.collisions_on:
            return None
        depth = self.maze.penetration(obs.footprint) if obs is not None else None
        event = self.monitor.update(t, depth, obs.center if obs is not None else None)
        if event is not None:
            self._add_event(event)
        return event

    def start(self, t: float, origin: Point | None = None) -> None:
        if self.state != WAITING:
            return
        self.state = RUNNING
        self.start_s = t
        self.now = t
        pos = origin or self.position
        if pos is not None:
            self.start_pos = pos
            self._advance(t, pos)

    def finish(self, t: float | None = None) -> ContactEvent | None:
        if self.state != RUNNING:
            return None
        self.now = self.now if t is None else t
        self.state = FINISHED
        self.end_s = self.now
        self.end_pos = self.position
        self.end_footprint = self.last_obs.footprint if self.last_obs is not None else None
        event = self.monitor.close()
        if event is not None:
            self._add_event(event)
        return event

    def toggle_collisions(self) -> bool:
        self.collisions_on = not self.collisions_on
        self.collisions_toggled = True
        if not self.collisions_on:
            event = self.monitor.close()
            if event is not None:
                self._add_event(event)
        return self.collisions_on

    def add_assist(self) -> None:
        self.assists.append(self.now)
        self._log.append(("assist", self.now))

    def add_judge_collision(self, severity: str) -> ContactEvent:
        x, y = self.position or (0.0, 0.0)
        event = ContactEvent(self.now, self.now, 0.0, x, y, severity, "judge")
        self._add_event(event)
        return event

    def undo_last(self) -> str | None:
        """Remove the most recent collision or assist, judge or camera logged."""
        if not self._log:
            return None
        kind, item = self._log.pop()
        if kind == "assist":
            self.assists.remove(item)
            return "assist"
        self.events.remove(item)
        return f"{item.severity} collision"

    def cycle_target(self) -> str | None:
        count = len(self.maze.drop_cells)
        if count == 0:
            return None
        if self.target_b is None:
            self.target_b = 0
        elif self.target_b + 1 >= count:
            self.target_b = None
        else:
            self.target_b += 1
        return self.target_name

    @property
    def target_name(self) -> str | None:
        return None if self.target_b is None else f"B{self.target_b + 1}"

    def collision_counts(self) -> dict[str, int]:
        return {
            "minor": sum(e.severity == "minor" for e in self.events),
            "major": sum(e.severity == "major" for e in self.events),
        }

    def drop_coverage(self) -> dict[str, float]:
        """Share of the robot footprint on each B cell at the end of the run."""
        footprint = self.end_footprint or (self.last_obs.footprint if self.last_obs else None)
        out = {}
        for i, cell in enumerate(self.maze.drop_cells):
            if footprint and len(footprint) >= 3:
                share = self.maze.coverage(footprint, cell)
            else:
                pos = self.end_pos or self.position
                share = 1.0 if pos is not None and self.maze.cell_at(pos) == cell else 0.0
            out[f"B{i + 1}"] = round(share, 3)
        return out

    def metrics(self) -> dict:
        """JSON friendly summary of the run for the scoresheet."""
        counts = self.collision_counts()
        lz = self.maze.loading_cells
        elapsed = self.elapsed_s
        end = self.end_pos or self.position
        return {
            "state": self.state,
            "elapsed_s": round(elapsed, 2),
            "time_limit_s": self.time_limit_s,
            "within_time_limit": self.state == FINISHED and elapsed <= self.time_limit_s,
            "start_ft": _round_point(self.start_pos),
            "start_cell": _cell_list(self.maze.cell_at(self.start_pos) if self.start_pos else None),
            "end_ft": _round_point(end),
            "end_cell": _cell_list(self.maze.cell_at(end) if end else None),
            "distance_ft": round(self.distance_ft, 2),
            "steps": self.steps,
            "unique_cells": len(self.cells_visited),
            "speed_steps_per_min": round(60 * self.steps / elapsed, 2) if elapsed > 0 else 0.0,
            "turns": self.turns,
            "drove": self.distance_ft >= DROVE_FT,
            "turned": self.turns > 0,
            "lz_reached_s": None if self.lz_reached_s is None else round(self.lz_reached_s, 2),
            "collision_tracking": self.collisions_on,
            "collision_tracking_toggled": self.collisions_toggled,
            "minor_collisions": counts["minor"],
            "major_collisions": counts["major"],
            "lz_collisions": sum(self.maze.cell_at((e.x_ft, e.y_ft)) in lz for e in self.events),
            "assists": len(self.assists),
            "collisions": [e.to_dict() for e in self.events],
            "assist_times_s": [round(self._rel(t), 2) for t in self.assists],
            "target_b": self.target_name,
            "b_coverage": self.drop_coverage(),
        }

    def _rel(self, t: float) -> float:
        return t - self.start_s if self.start_s is not None else t

    def _add_event(self, event: ContactEvent) -> None:
        self.events.append(event)
        self._log.append(("collision", event))

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
            if cell not in self.maze.loading_cells:
                self._outside_lz = True
            elif self._outside_lz and self.lz_reached_s is None:
                self.lz_reached_s = self._rel(t)
        if self._heading_anchor is None:
            self._heading_anchor = p
        elif math.dist(self._heading_anchor, p) >= TURN_SEGMENT_FT:
            heading = math.atan2(p[1] - self._heading_anchor[1], p[0] - self._heading_anchor[0])
            if self._heading is not None and _angle_between(heading, self._heading) >= TURN_DEG:
                self.turns += 1
            self._heading = heading
            self._heading_anchor = p


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


def _mix(a: Point, b: Point, w: float) -> Point:
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def _angle_between(a: float, b: float) -> float:
    diff = math.degrees(abs(a - b)) % 360
    return min(diff, 360 - diff)


def _round_point(p: Point | None) -> list[float] | None:
    return None if p is None else [round(p[0], 3), round(p[1], 3)]


def _cell_list(cell: Cell | None) -> list[int] | None:
    return None if cell is None else [cell[0], cell[1]]
