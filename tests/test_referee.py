import math

import pytest

from utra_tracker.config import GradingConfig
from utra_tracker.maze import Maze
from utra_tracker.referee import FINISHED, RUNNING, WAITING, ContactMonitor, Observation, Referee, pick_robot

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


def test_auto_start_records_start_where_robot_was_placed():
    ref = referee()
    d = Driver(ref)
    d.hold(3.5, 1.5, 10)
    assert ref.state == WAITING
    d.drive((3.5, 0.5))
    assert ref.state == RUNNING
    assert ref.start_pos == pytest.approx((3.5, 1.5), abs=0.01)


def test_no_auto_start_waits_for_space():
    ref = referee(auto_start=False)
    d = Driver(ref)
    d.hold(3.5, 1.5, 3)
    d.drive((3.5, 0.5))
    assert ref.state == WAITING
    ref.start(d.t)
    assert ref.state == RUNNING


def test_full_run_metrics():
    ref = referee()
    d = Driver(ref)
    d.hold(3.5, 1.5, 5)
    d.drive((3.5, 0.5), (1.5, 0.5), (3.5, 0.5), (3.5, 1.5), (7.5, 1.5), (7.5, 3.5))
    d.hold(7.5, 3.5, 10)
    ref.target_b = 3
    ref.finish(d.t)
    m = ref.metrics()
    assert ref.state == FINISHED
    assert m["steps"] == 12
    # Smoothing rounds the corners a little, the rubric counts 1 ft steps anyway.
    assert m["distance_ft"] == pytest.approx(12.0, abs=0.8)
    assert m["turns"] >= 4 and m["drove"] and m["turned"]
    assert m["lz_reached_s"] is not None
    assert m["minor_collisions"] == m["major_collisions"] == 0
    assert m["end_cell"] == [7, 3]
    assert m["b_coverage"]["B4"] == pytest.approx(1.0)
    assert m["within_time_limit"]
    assert m["speed_steps_per_min"] > 0


def test_starting_inside_loading_zone_is_not_an_arrival():
    ref = referee()
    d = Driver(ref)
    d.hold(0.5, 0.5, 3)
    d.drive((1.5, 0.5))
    assert ref.lz_reached_s is None
    d.drive((2.5, 0.5), (1.5, 0.5))
    assert ref.lz_reached_s is not None


def test_contact_is_debounced_and_graded():
    ref = referee(contact_frames=3, release_frames=3, major_duration_s=1.0, major_depth_ft=0.15)
    d = Driver(ref)
    d.hold(4.5, 1.5, 3)
    ref.start(d.t)
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
    d.hold(4.5, 1.5, 2)
    ref.start(d.t)
    assert ref.toggle_collisions() is False
    d.hold(4.5, 1.2, 20)
    d.hold(4.5, 1.5, 3)
    assert ref.events == []
    assert ref.toggle_collisions() is True
    d.hold(4.5, 1.2, 3)
    d.hold(4.5, 1.5, 3)
    assert len(ref.events) == 1
    assert ref.metrics()["collision_tracking_toggled"]


def test_judge_logs_and_undo():
    ref = referee()
    ref.start(0.0)
    ref.add_judge_collision("minor")
    ref.add_assist()
    ref.add_judge_collision("major")
    assert ref.collision_counts() == {"minor": 1, "major": 1}
    assert ref.undo_last() == "major collision"
    assert ref.undo_last() == "assist"
    assert ref.assists == []
    assert ref.undo_last() == "minor collision"
    assert ref.undo_last() is None


def test_cycle_target_wraps_to_none():
    ref = referee()
    names = [ref.cycle_target() for _ in range(5)]
    assert names == ["B1", "B2", "B3", "B4", None]


def test_time_limit_follows_milestone():
    assert referee(milestone=2).time_limit_s == 300
    assert referee(milestone=3).time_limit_s == 480
    assert referee(milestone=3, time_limit_s=60.0).time_limit_s == 60


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
