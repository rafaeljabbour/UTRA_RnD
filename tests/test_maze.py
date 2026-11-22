import math

import pytest

from utra_tracker.maze import (
    DEFAULT_LAYOUT,
    Homography,
    Maze,
    MazeError,
    correct_parallax,
    pixel_to_maze,
    point_in_polygon,
    polygon_centroid,
)


def square(cx, cy, half):
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


def test_default_layout_matches_rubric_figure():
    maze = Maze()
    assert (maze.cols, maze.rows) == (8, 4)
    assert (maze.width_ft, maze.height_ft) == (8.0, 4.0)
    blocks = {(c, r) for r in range(4) for c in range(8) if not maze.is_free((c, r))}
    assert blocks == {(4, 0), (6, 0), (2, 1), (1, 2), (3, 2), (4, 2), (6, 2), (6, 3)}
    assert maze.loading_cells == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert maze.drop_cells == ((5, 0), (7, 0), (2, 2), (7, 3))


def test_walls_are_block_faces_and_border():
    segs = Maze().wall_segments()
    # Outer border is 2 * (8 + 4) = 24 unit edges, minus those covered by blocks on the border,
    # plus the inner block faces. Spot check a few instead of the exact count.
    assert ((0.0, 0.0), (1.0, 0.0)) in segs  # top border at the loading zone
    assert ((4.0, 1.0), (5.0, 1.0)) in segs  # bottom face of block (4, 0)
    assert ((2.0, 2.0), (2.0, 3.0)) in segs  # right face of block (1, 2), left side of B3
    assert ((5.0, 1.0), (6.0, 1.0)) not in segs  # open floor between B1 and the cell below


def test_cells_and_depth():
    maze = Maze()
    assert maze.cell_at((0.5, 0.5)) == (0, 0)
    assert maze.cell_at((-0.1, 0.5)) is None
    assert maze.depth((3.5, 1.5)) == 0.0
    assert maze.depth((4.5, 0.9)) == pytest.approx(0.1)  # inside block (4, 0), 0.1 ft above its face
    assert maze.depth((8.2, 1.5)) == pytest.approx(0.2)  # outside the right wall
    # Point inside the 2 ft block (3,2)-(4,2) near their shared edge is deep, not shallow.
    assert maze.depth((4.0, 2.5)) == pytest.approx(0.5)


def test_penetration_of_footprints():
    maze = Maze()
    assert maze.penetration(square(3.5, 1.5, 0.4)) == 0.0
    assert maze.penetration(square(4.5, 1.35, 0.4)) == pytest.approx(0.05, abs=0.01)
    assert maze.penetration(square(0.35, 3.5, 0.4)) == pytest.approx(0.05, abs=0.01)


def test_inset_cell_hysteresis():
    maze = Maze()
    assert maze.inset_cell((3.5, 1.5), 0.1) == (3, 1)
    assert maze.inset_cell((3.95, 1.5), 0.1) is None
    assert maze.inset_cell((4.5, 0.5), 0.1) is None  # block


def test_coverage():
    maze = Maze()
    assert maze.coverage(square(7.5, 3.5, 0.3), (7, 3)) == pytest.approx(1.0)
    half = maze.coverage(square(7.0, 3.5, 0.3), (7, 3))
    assert 0.4 < half < 0.6
    assert maze.coverage(square(2.5, 0.5, 0.3), (7, 3)) == 0.0


@pytest.mark.parametrize(
    "layout",
    [[], ["LL", "L"], ["LX"], ["##", "##"], "LL..", [1, 2]],
)
def test_bad_layouts(layout):
    with pytest.raises(MazeError):
        Maze(tuple(layout) if isinstance(layout, list) else layout)


def test_homography_roundtrip_on_tilted_quad():
    maze = Maze(DEFAULT_LAYOUT)
    corners = [(140.0, 120.0), (1150.0, 95.0), (1170.0, 610.0), (120.0, 630.0)]
    h = pixel_to_maze(corners, maze)
    for px, ft in zip(corners, [(0, 0), (8, 0), (8, 4), (0, 4)], strict=True):
        got = h.apply(px)
        assert got == pytest.approx(ft, abs=1e-6)
    inv = h.inverse()
    for p in [(1.5, 0.5), (7.2, 3.9), (4.0, 2.0)]:
        assert h.apply(inv.apply(p)) == pytest.approx(p, abs=1e-6)


def test_homography_rejects_degenerate_points():
    with pytest.raises(MazeError):
        Homography.from_points([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (8, 0), (8, 4), (0, 4)])


def test_parallax_pulls_points_toward_nadir():
    p = correct_parallax((8.0, 2.0), (4.0, 2.0), camera_h=8.0, object_h=0.5)
    assert p == pytest.approx((4.0 + 4.0 * 7.5 / 8.0, 2.0))


def test_polygon_helpers():
    sq = square(2.0, 1.0, 0.5)
    assert polygon_centroid(sq) == pytest.approx((2.0, 1.0))
    assert point_in_polygon((2.0, 1.0), sq)
    assert not point_in_polygon((3.0, 1.0), sq)
    circle = [(math.cos(a / 10 * math.tau), math.sin(a / 10 * math.tau)) for a in range(10)]
    assert polygon_centroid(circle) == pytest.approx((0.0, 0.0), abs=1e-9)
