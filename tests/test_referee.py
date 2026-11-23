import math

import pytest

from utra_tracker.config import GradingConfig
from utra_tracker.maze import Maze
from utra_tracker.referee import (
    FAIL,
    FINISHED,
    PASS,
    REACHED_END,
    RUNNING,
    STOPPED,
    TIME_LIMIT,
    WAITING,
    ContactMonitor,
    Observation,
    Referee,
    pick_robot,
)

FPS = 10


def obs(x, y, r=0.35, conf=0.8):
    pts = tuple((x + r * math.cos(a * math.tau / 16), y + r * math.sin(a * math.tau / 16)) for a in range(16))
    return Observation((x, y), pts, conf)


class Driver:
    """Feeds a referee a straight line route at a fixed speed."""

    def __init__(self, referee):
        self.ref = referee
        self.t = 0.0
        self.pos = None

    def hold(self, x, y, frames):
        for _ in range(frames):
            self.ref.update(self.t, obs(x, y))
            self.pos = (x, y)
            self.t += 1 / FPS

    def drive(self, *points, speed=1.0):
        for target in points:
            start = self.pos
            n = max(1, round(math.dist(start, target) / speed * FPS))
            for k in range(1, n + 1):
                p = (start[0] + (target[0] - start[0]) * k / n, start[1] + (target[1] - start[1]) * k / n)
                self.ref.update(self.t, obs(*p))
                self.t += 1 / FPS
            self.pos = target


def referee(**kw):
    return Referee(Maze(), GradingConfig(**kw))


ROUTE_TO_B4 = ((2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (7.5, 1.5), (7.5, 3.5))


def test_clock_starts_when_robot_moves():
    ref = referee()
    d = Driver(ref)
    d.hold(1.0, 1.0, 10)
    assert ref.state == WAITING
    d.drive((2.5, 0.5))
    assert ref.state == RUNNING
    assert ref.start_pos == pytest.approx((1.0, 1.0), abs=0.01)
    assert ref.started_in_zone


def test_pass_from_loading_zone_to_b_finishes_by_itself():
    ref = referee(settle_s=2.0)
    d = Driver(ref)
    d.hold(1.0, 1.0, 5)
    d.drive(*ROUTE_TO_B4)
    assert ref.state == RUNNING
    d.hold(7.5, 3.5, 25)
    assert ref.state == FINISHED
    assert ref.finish_reason == REACHED_END
    assert ref.outcome == PASS
    m = ref.metrics()
    assert m["start_cell"] == [1, 1] and m["end_cell"] == [7, 3]
    assert m["end_zone_share"] == pytest.approx(1.0)
    assert m["reached_end_zone_s"] is not None
    assert m["steps"] == 9  # (1,0) to (3,0), down to (3,1), across to (7,1), down to (7,3)
    # Smoothing rounds the corners a little.
    assert m["distance_ft"] == pytest.approx(10.0, abs=0.8)


def test_passing_through_an_end_cell_does_not_finish():
    ref = referee(settle_s=2.0)
    d = Driver(ref)
    d.hold(1.0, 1.0, 5)
    d.drive((2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (5.5, 1.5), (5.5, 0.5), (5.5, 1.5))
    assert ref.state == RUNNING
    assert ref.reached_end_s is not None


def test_fail_when_started_outside_start_zone():
    ref = referee()
    d = Driver(ref)
    d.hold(3.5, 1.5, 5)
    d.drive((3.5, 1.5 - 0.01), (7.5, 1.5), (7.5, 3.5))
    d.hold(7.5, 3.5, 30)
    assert ref.state == FINISHED
    assert not ref.started_in_zone and ref.ended_in_zone
    assert ref.outcome == FAIL


def test_time_limit_ends_run_as_fail():
    ref = referee(time_limit_s=3.0)
    d = Driver(ref)
    d.hold(1.0, 1.0, 5)
    d.drive((2.5, 0.5), (3.5, 0.5))
    d.hold(3.5, 0.5, 30)
    assert ref.state == FINISHED and ref.finish_reason == TIME_LIMIT
    assert ref.outcome == FAIL
    assert ref.elapsed_s == pytest.approx(3.0, abs=0.15)


def test_no_time_limit():
    ref = referee(time_limit_s=None)
    d = Driver(ref)
    d.hold(1.0, 1.0, 3)
    d.drive((2.5, 0.5))
    d.hold(2.5, 0.5, 3000)
    assert ref.state == RUNNING


def test_stopping_early_uses_last_position():
    ref = referee()
    d = Driver(ref)
    d.hold(1.0, 1.0, 3)
    d.drive((2.5, 0.5), (3.5, 0.5))
    ref.finish(d.t)
    assert ref.finish_reason == STOPPED and ref.outcome == FAIL
    assert ref.metrics()["end_cell"] == [3, 0]


def test_partly_in_end_zone_is_not_enough():
    ref = referee()
    d = Driver(ref)
    d.hold(1.0, 1.0, 3)
    d.drive((2.5, 0.5), (3.5, 0.5), (3.5, 1.5), (7.5, 1.5), (7.5, 2.9))
    d.hold(7.5, 2.9, 30)
    assert ref.state == RUNNING
    ref.finish(d.t)
    assert 0 < ref.end_share < 0.5 and ref.outcome == FAIL


def test_custom_zones_from_cells():
    ref = referee(start_zone=((3, 1),), end_zone=((5, 1),))
    d = Driver(ref)
    d.hold(3.5, 1.5, 3)
    d.drive((5.5, 1.5))
    d.hold(5.5, 1.5, 30)
    assert ref.outcome == PASS


def test_contact_is_debounced_and_graded():
    ref = referee(contact_frames=3, release_frames=3, major_duration_s=1.0, major_depth_ft=0.15)
    d = Driver(ref)
    d.hold(3.5, 1.5, 3)
    d.drive((4.5, 1.5))
    assert ref.state == RUNNING
    # Block (4, 0) ends at y = 1, so a 0.35 ft radius robot at y = 1.30 is 0.05 ft into it.
    d.hold(4.5, 1.30, 2)  # only 2 frames: ignored
    d.hold(4.5, 1.5, 5)
    assert ref.events == []
    d.hold(4.5, 1.30, 5)  # short and shallow: minor
    d.hold(4.5, 1.5, 5)
    d.hold(4.5, 1.30, 15)  # long: major
    d.hold(4.5, 1.5, 5)
    d.hold(4.5, 1.15, 4)  # 0.2 ft deep: major
    ref.finish(d.t)
    assert [e.severity for e in ref.events] == ["minor", "major", "major"]


def test_collision_toggle_stops_camera_events():
    ref = referee(contact_frames=1, release_frames=1)
    d = Driver(ref)
    d.hold(3.5, 1.5, 3)
    d.drive((4.5, 1.5))
    assert ref.toggle_collisions() is False
    d.hold(4.5, 1.2, 20)
    d.hold(4.5, 1.5, 3)
    assert ref.events == []
    assert ref.toggle_collisions() is True
    d.hold(4.5, 1.2, 3)
    d.hold(4.5, 1.5, 3)
    assert len(ref.events) == 1
    m = ref.metrics()
    assert m["collision_tracking_toggled"] and m["major_collisions"] + m["minor_collisions"] == 1


def test_reset_starts_over():
    ref = referee()
    d = Driver(ref)
    d.hold(1.0, 1.0, 3)
    d.drive((2.5, 0.5))
    ref.reset()
    assert ref.state == WAITING and ref.path == [] and ref.start_pos is None


def test_monitor_flushes_open_contact():
    mon = ContactMonitor(GradingConfig(contact_frames=1))
    mon.update(0.0, 0.2, (1.0, 1.0))
    assert mon.in_contact
    event = mon.close()
    assert event is not None and event.severity == "major"
    assert mon.close() is None


def test_pick_robot_prefers_inside_and_nearby():
    maze = Maze()
    outside = obs(12.0, 2.0, conf=0.95)
    far = obs(7.5, 3.5, conf=0.6)
    near = obs(3.6, 1.5, conf=0.5)
    assert pick_robot([outside], maze, None) is None
    assert pick_robot([outside, far, near], maze, None) is far
    assert pick_robot([far, near], maze, (3.5, 1.5)) is near
