import pytest

from tests_support import finished_metrics as metrics
from utra_tracker.rubric import (
    BASE_TOTAL,
    RUBRICS,
    build_sheet,
    format_sheet,
    load_sheet,
    rank_speed,
    save_sheet,
    suggest,
    totals,
)


@pytest.mark.parametrize(("milestone", "total"), [(2, 20), (3, 60)])
def test_rubric_maxima_match_pdf(milestone, total):
    assert sum(i.max for i in RUBRICS[milestone] if not i.bonus) == total == BASE_TOTAL[milestone]


@pytest.mark.parametrize(
    ("minor", "major", "assists", "expected"),
    [(0, 0, 0, 8), (1, 0, 0, 7), (0, 1, 0, 6), (0, 0, 1, 5), (1, 1, 0, 5), (3, 2, 1, 3), (10, 5, 5, 2)],
)
def test_m3_obstacle_avoidance(minor, major, assists, expected):
    m = metrics(minor_collisions=minor, major_collisions=major, assists=assists)
    assert suggest(3, m)["no_contact"][0] == expected


@pytest.mark.parametrize(
    ("minor", "major", "expected"), [(0, 0, 2), (1, 0, 1.5), (0, 1, 1), (2, 0, 0.75), (9, 0, 0)]
)
def test_m2_obstacle_avoidance(minor, major, expected):
    assert suggest(2, metrics(minor_collisions=minor, major_collisions=major))["no_contact"][0] == expected


def test_functional_needs_drive_and_turn():
    assert suggest(3, metrics())["functional"][0] == 8
    assert suggest(3, metrics(turned=False))["functional"][0] == 4
    assert suggest(2, metrics(drove=False, turned=False))["functional"][0] == 0


@pytest.mark.parametrize(
    ("target", "coverage", "expected"),
    [("B2", 0.8, 4), ("B2", 0.3, 2), ("B2", 0.0, 0), (None, 0.6, 4)],
)
def test_finish_in_b(target, coverage, expected):
    m = metrics(target_b=target, b_coverage={"B1": 0.0, "B2": coverage})
    assert suggest(3, m)["finish_b"][0] == expected


def test_lz_and_continuous():
    s = suggest(3, metrics())
    assert s["lz_confirm"][0] == 3 and s["continuous"][0] == 5
    s = suggest(3, metrics(lz_reached_s=None))
    assert s["lz_confirm"][0] == 0 and "continuous" not in s
    assert suggest(3, metrics(assists=2))["continuous"][0] is None


def test_judge_overrides_and_line_of_sight_cap(tmp_path):
    sheet = build_sheet(3, "7", "1", metrics())
    t = totals(sheet)
    assert t["total"] == 8 + 8 + 3 + 4 + 5
    assert "build_quality" in t["pending"]
    sheet["items"]["no_contact"]["judge"] = 6
    sheet["items"]["bonus"]["judge"] = 2
    assert totals(sheet)["total"] == 8 + 6 + 3 + 4 + 5 + 2
    sheet["control"] = "line_of_sight"
    assert totals(sheet)["total"] == pytest.approx(0.7 * 28)
    path = tmp_path / "scoresheet.yaml"
    save_sheet(path, sheet)
    again = load_sheet(tmp_path)
    assert again["items"]["no_contact"]["judge"] == 6
    assert "x0.7 line of sight" in format_sheet(again)


def test_load_sheet_rejects_other_yaml(tmp_path):
    path = tmp_path / "scoresheet.yaml"
    path.write_text("milestone: 9\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sheet(path)


def test_rank_speed():
    fast = build_sheet(2, "1", "1", metrics(speed_steps_per_min=20.0))
    mid = build_sheet(2, "2", "1", metrics(speed_steps_per_min=10.0))
    slow_late = build_sheet(2, "3", "1", metrics(speed_steps_per_min=5.0, within_time_limit=False))
    unfinished = build_sheet(2, "4", "1", metrics(state="running"))
    points = {s["team"]: p for s, p in rank_speed([slow_late, mid, unfinished, fast])}
    assert points == {"1": 4.0, "2": 3.0, "3": 0.0, "4": 0.0}
    m3 = {s["team"]: p for s, p in rank_speed([build_sheet(3, "9", "1", metrics())])}
    assert m3 == {"9": 5.0}
