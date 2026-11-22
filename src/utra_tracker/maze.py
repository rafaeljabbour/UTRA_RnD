"""Maze layout, pixel to maze mapping and wall contact geometry.

Everything here is plain Python so it can be tested without numpy or OpenCV.
Maze coordinates are in feet with the origin at the outer corner of the loading zone,
x along the long 8 ft side and y along the short 4 ft side.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

Point = tuple[float, float]
Cell = tuple[int, int]

FREE, BLOCK, LOAD, DROP = ".", "#", "L", "B"
CELL_CHARS = {FREE, BLOCK, LOAD, DROP}

# MIE444 maze from the Milestone 2 and 3 rubrics. Each character is one 1 ft square.
# Row 0 is the side with the loading zone. Every wall in the rubric figure is the face of a block.
DEFAULT_LAYOUT = (
    "LL..#B#B",
    "LL#.....",
    ".#B##.#.",
    "......#B",
)


class MazeError(ValueError):
    """Raised for an invalid layout or calibration."""


@dataclass(frozen=True)
class Maze:
    layout: tuple[str, ...] = DEFAULT_LAYOUT
    cell_ft: float = 1.0

    def __post_init__(self) -> None:
        validate_layout(self.layout)
        if not (isinstance(self.cell_ft, (int, float)) and self.cell_ft > 0):
            raise MazeError("cell size must be positive")

    @property
    def rows(self) -> int:
        return len(self.layout)

    @property
    def cols(self) -> int:
        return len(self.layout[0])

    @property
    def width_ft(self) -> float:
        return self.cols * self.cell_ft

    @property
    def height_ft(self) -> float:
        return self.rows * self.cell_ft

    def kind(self, cell: Cell) -> str:
        col, row = cell
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return self.layout[row][col]
        return BLOCK

    def is_free(self, cell: Cell) -> bool:
        return self.kind(cell) != BLOCK

    @cached_property
    def free_cells(self) -> tuple[Cell, ...]:
        return tuple((c, r) for r in range(self.rows) for c in range(self.cols) if self.layout[r][c] != BLOCK)

    @cached_property
    def loading_cells(self) -> frozenset[Cell]:
        return frozenset(c for c in self.free_cells if self.kind(c) == LOAD)

    @cached_property
    def drop_cells(self) -> tuple[Cell, ...]:
        """Drop off cells named B1, B2 and so on in reading order."""
        return tuple(c for c in self.free_cells if self.kind(c) == DROP)

    def cell_at(self, p: Point) -> Cell | None:
        col, row = math.floor(p[0] / self.cell_ft), math.floor(p[1] / self.cell_ft)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return (col, row)
        return None

    def cell_rect(self, cell: Cell) -> tuple[float, float, float, float]:
        s = self.cell_ft
        return (cell[0] * s, cell[1] * s, (cell[0] + 1) * s, (cell[1] + 1) * s)

    def cell_center(self, cell: Cell) -> Point:
        s = self.cell_ft
        return ((cell[0] + 0.5) * s, (cell[1] + 0.5) * s)

    def inset_cell(self, p: Point, inset_ft: float) -> Cell | None:
        """The free cell holding p, only if p is at least inset_ft away from its edges."""
        cell = self.cell_at(p)
        if cell is None or not self.is_free(cell):
            return None
        x0, y0, x1, y1 = self.cell_rect(cell)
        if min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) < inset_ft:
            return None
        return cell

    def depth(self, p: Point) -> float:
        """How far p is inside a block or outside the arena, 0 when it is in open floor."""
        cell = self.cell_at(p)
        if cell is not None and self.is_free(cell):
            return 0.0
        return min(_rect_distance(p, self.cell_rect(c)) for c in self.free_cells)

    def penetration(self, footprint: Sequence[Point], step_ft: float = 0.04) -> float:
        """Deepest point of a robot outline inside any obstacle, in feet."""
        return max((self.depth(p) for p in sample_outline(footprint, step_ft)), default=0.0)

    def coverage(self, footprint: Sequence[Point], cell: Cell, step_ft: float = 0.04) -> float:
        """Fraction of the footprint area that lies inside cell, by grid sampling."""
        if len(footprint) < 3:
            return 0.0
        xs, ys = [p[0] for p in footprint], [p[1] for p in footprint]
        x0, y0, x1, y1 = self.cell_rect(cell)
        if max(xs) <= x0 or min(xs) >= x1 or max(ys) <= y0 or min(ys) >= y1:
            return 0.0
        inside = total = 0
        y = min(ys) + step_ft / 2
        while y < max(ys):
            x = min(xs) + step_ft / 2
            while x < max(xs):
                if point_in_polygon((x, y), footprint):
                    total += 1
                    if x0 <= x < x1 and y0 <= y < y1:
                        inside += 1
                x += step_ft
            y += step_ft
        return inside / total if total else 0.0

    def wall_segments(self) -> list[tuple[Point, Point]]:
        """Edges between open floor and a block or the outside, for drawing."""
        s = self.cell_ft
        segs = []
        for c, r in self.free_cells:
            if not self.is_free((c, r - 1)):
                segs.append(((c * s, r * s), ((c + 1) * s, r * s)))
            if not self.is_free((c, r + 1)):
                segs.append(((c * s, (r + 1) * s), ((c + 1) * s, (r + 1) * s)))
            if not self.is_free((c - 1, r)):
                segs.append(((c * s, r * s), (c * s, (r + 1) * s)))
            if not self.is_free((c + 1, r)):
                segs.append((((c + 1) * s, r * s), ((c + 1) * s, (r + 1) * s)))
        return segs


def validate_layout(layout: Sequence[str]) -> None:
    if not isinstance(layout, (list, tuple)) or not layout:
        raise MazeError("arena.layout must be a non empty list of rows")
    if not all(isinstance(row, str) and row for row in layout):
        raise MazeError("arena.layout rows must be non empty strings")
    if len({len(row) for row in layout}) != 1:
        raise MazeError("arena.layout rows must all be the same length")
    bad = {ch for row in layout for ch in row} - CELL_CHARS
    if bad:
        raise MazeError(f"arena.layout has unknown cell characters {sorted(bad)}, use . # L B")
    if not any(ch != BLOCK for row in layout for ch in row):
        raise MazeError("arena.layout has no open cells")


@dataclass(frozen=True)
class Homography:
    """3x3 projective map stored row major."""

    m: tuple[float, ...]

    @classmethod
    def from_points(cls, src: Sequence[Point], dst: Sequence[Point]) -> Homography:
        if len(src) != 4 or len(dst) != 4:
            raise MazeError("a homography needs exactly 4 point pairs")
        a, b = [], []
        for (x, y), (u, v) in zip(src, dst, strict=True):
            a.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            b.append(u)
            a.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            b.append(v)
        h = _solve(a, b)
        return cls((*h, 1.0))

    @classmethod
    def identity(cls) -> Homography:
        return cls((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    def apply(self, p: Point) -> Point:
        m = self.m
        x, y = p
        w = m[6] * x + m[7] * y + m[8]
        if abs(w) < 1e-12:
            w = 1e-12
        return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)

    def inverse(self) -> Homography:
        a, b, c, d, e, f, g, h, i = self.m
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(det) < 1e-12:
            raise MazeError("calibration corners are degenerate")
        adj = (
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        )
        return Homography(tuple(v / det for v in adj))


def maze_corners(maze: Maze) -> list[Point]:
    """Arena corners in click order: loading zone corner, then clockwise along the long side."""
    w, h = maze.width_ft, maze.height_ft
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def pixel_to_maze(corners_px: Sequence[Point], maze: Maze) -> Homography:
    return Homography.from_points(corners_px, maze_corners(maze))


def correct_parallax(p: Point, nadir: Point, camera_h: float, object_h: float) -> Point:
    """Move a point seen at object_h above the floor to the floor point directly below it."""
    k = (camera_h - object_h) / camera_h
    return (nadir[0] + (p[0] - nadir[0]) * k, nadir[1] + (p[1] - nadir[1]) * k)


def sample_outline(poly: Sequence[Point], step_ft: float) -> list[Point]:
    if len(poly) < 2:
        return list(poly)
    pts = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        n = max(1, math.ceil(math.dist(a, b) / step_ft))
        pts.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n))
    return pts


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly: Sequence[Point]) -> Point:
    area = cx = cy = 0.0
    for i, (x0, y0) in enumerate(poly):
        x1, y1 = poly[(i + 1) % len(poly)]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-12:
        return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    return (cx / (3 * area), cy / (3 * area))


def _rect_distance(p: Point, rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return math.hypot(dx, dy)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    rows = [[*row, rhs] for row, rhs in zip(a, b, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise MazeError("calibration corners are degenerate, click 4 distinct corners")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            f = rows[r][col] / rows[col][col]
            for k in range(col, n + 1):
                rows[r][k] -= f * rows[col][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (rows[r][n] - sum(rows[r][k] * x[k] for k in range(r + 1, n))) / rows[r][r]
    return x
