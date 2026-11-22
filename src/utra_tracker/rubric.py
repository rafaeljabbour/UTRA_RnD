"""MIE444 milestone rubrics, score suggestions from run metrics, scoresheets and speed ranking.

The camera can only judge what it sees: movement, wall contact, where the rover went and how fast.
Those items get a suggested score. Everything else, such as build quality or the pick and place, is
left for the judge. A judge score always wins over a suggestion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SHEET_NAME = "scoresheet.yaml"
LINE_OF_SIGHT_FACTOR = 0.7
CONTROL_MODES = ("autonomous", "partial", "sensors_only", "line_of_sight")


@dataclass(frozen=True)
class Item:
    key: str
    section: str
    label: str
    max: float
    hint: str = ""
    bonus: bool = False


MILESTONE_2 = (
    Item("functional", "Obstacle Avoidance", "Functional rover that can drive and turn", 4),
    Item(
        "build_quality",
        "Obstacle Avoidance",
        "Build quality of the rover",
        2,
        "professional/good/acceptable/poor",
    ),
    Item("no_contact", "Obstacle Avoidance", "Travels in loading zone without wall contact", 2),
    Item(
        "speed", "Obstacle Avoidance", "Speed, ranked fastest to slowest", 4, "set by utra-track leaderboard"
    ),
    Item(
        "finds_load",
        "Block Pick and Place",
        "Finds and drives up to the load",
        3,
        "random 3, random four 2, team spot 1",
    ),
    Item("picks_load", "Block Pick and Place", "Picks up the load", 3),
    Item("places_load", "Block Pick and Place", "Places down the load outside the loading zone", 2),
    Item("bonus", "Bonus", "Sensor data only or full automation", 2, "full auto 2, partial 1", bonus=True),
    Item("usb_deduction", "Deductions", "USB cable needed to complete the run", 0, "enter -1 if tethered"),
)

MILESTONE_3 = (
    Item("functional", "Obstacle Avoidance", "Functional robot that can drive and turn", 8),
    Item("build_quality", "Obstacle Avoidance", "Build quality of the rover", 4, "strict, few teams get 4"),
    Item("no_contact", "Obstacle Avoidance", "Completes the run without wall contact", 8),
    Item(
        "localization",
        "Localization",
        "Achieves localization",
        5,
        "random 5, wall aligned 3, team specified 0",
    ),
    Item(
        "lz_confirm",
        "Localization",
        "Arrives at loading zone with confirmation",
        3,
        "LED, screen text or heatmap",
    ),
    Item("finish_b", "Localization", "Arrives and finishes in location B", 4),
    Item("pnp_mechanism", "Load Pick and Place", "Functional pick and place mechanism", 4),
    Item(
        "finds_load",
        "Load Pick and Place",
        "Finds the load using sensors",
        4,
        "random 4, random four 3, team spot 1",
    ),
    Item("drives_to_load", "Load Pick and Place", "Drives up to the load", 2),
    Item("picks_load", "Load Pick and Place", "Picks up the load", 4),
    Item("delivers_load", "Load Pick and Place", "Delivers the load to location B", 2),
    Item("places_load", "Load Pick and Place", "Places down the load", 2),
    Item("speed", "Integration", "Speed, ranked fastest to slowest", 5, "set by utra-track leaderboard"),
    Item(
        "continuous",
        "Integration",
        "Tasks in one continuous run without assistance",
        5,
        "continuous 5, one task alone 3, two alone 1, all alone 0",
    ),
    Item("bonus", "Bonus", "Complete automation, 1 per autonomous part", 4, bonus=True),
)

RUBRICS = {2: MILESTONE_2, 3: MILESTONE_3}
BASE_TOTAL = {2: 20, 3: 60}
SPEED_MAX = {2: 4.0, 3: 5.0}


def items_for(milestone: int) -> tuple[Item, ...]:
    if milestone not in RUBRICS:
        raise ValueError(f"no rubric for milestone {milestone}")
    return RUBRICS[milestone]


def suggest(milestone: int, m: dict[str, Any]) -> dict[str, tuple[float | None, str]]:
    """Suggested score and a short reason for each item the camera can judge."""
    out: dict[str, tuple[float | None, str]] = {}
    full = items_for(milestone)[0].max
    moved = f"{m['distance_ft']:.1f} ft, {m['steps']} steps, {m['turns']} turns"
    if m["drove"] and m["turned"]:
        out["functional"] = (full, f"drove and turned: {moved}")
    elif m["drove"]:
        out["functional"] = (full / 2, f"drove but no turn seen: {moved}")
    else:
        out["functional"] = (0.0, f"no driving seen: {moved}")

    out["no_contact"] = _no_contact(milestone, m)
    out["speed"] = (None, f"{m['speed_steps_per_min']} steps/min, rank with utra-track leaderboard")

    if milestone == 2:
        out["usb_deduction"] = (0.0, "enter -1 if a USB cable was needed")
    if milestone == 3:
        if m["lz_reached_s"] is not None:
            out["lz_confirm"] = (3.0, f"entered loading zone at {m['lz_reached_s']:.1f} s, check the signal")
        else:
            out["lz_confirm"] = (0.0, "never entered the loading zone")
        out["finish_b"] = _finish_b(m)
        if m["assists"] == 0 and m["lz_reached_s"] is not None and out["finish_b"][0]:
            out["continuous"] = (5.0, "no assistance, reached loading zone and B")
        elif m["assists"]:
            out["continuous"] = (None, f"{m['assists']} assists logged, judge decides")
    return out


def _no_contact(milestone: int, m: dict[str, Any]) -> tuple[float | None, str]:
    minor, major, assists = m["minor_collisions"], m["major_collisions"], m["assists"]
    tally = f"{minor} minor, {major} major, {assists} assists"
    if not m["collision_tracking"] and not (minor or major):
        tally += ", camera tracking off"
    events = minor + major + assists
    if milestone == 2:
        if events == 0:
            return 2.0, f"perfect run: {tally}"
        if events == 1 and minor:
            return 1.5, f"one minor contact: {tally}"
        if events == 1:
            return 1.0, f"one major contact or assist: {tally}"
        return max(0.0, 1.0 - 0.25 * (events - 1)), f"multiple, ranked against teams: {tally}"
    if events == 0:
        return 8.0, f"perfect run: {tally}"
    if events == 1:
        score = 7.0 if minor else 6.0 if major else 5.0
        return score, tally
    return max(2.0, 5.0 - 0.5 * (events - 2)), f"multiple, ranked against teams: {tally}"


def _finish_b(m: dict[str, Any]) -> tuple[float | None, str]:
    coverage: dict[str, float] = m["b_coverage"]
    if not coverage:
        return None, "layout has no B cells"
    target = m["target_b"]
    if target is None:
        target = max(coverage, key=coverage.get)
        if coverage[target] == 0:
            return 0.0, "ended outside every B, no target set"
        prefix = f"no target set, nearest {target}"
    else:
        prefix = f"target {target}"
    share = coverage.get(target, 0.0)
    if share >= 0.5:
        return 4.0, f"{prefix}: {share:.0%} of rover inside"
    if share > 0:
        return 2.0, f"{prefix}: only partly inside ({share:.0%})"
    return 0.0, f"{prefix}: rover not in B"


def build_sheet(milestone: int, team: str, run: str, metrics: dict[str, Any]) -> dict[str, Any]:
    suggestions = suggest(milestone, metrics)
    items = {}
    for item in items_for(milestone):
        auto, note = suggestions.get(item.key, (None, ""))
        items[item.key] = {
            "label": item.label,
            "max": item.max,
            "auto": auto,
            "judge": None,
            "note": note or item.hint,
        }
    sheet = {
        "milestone": milestone,
        "team": team,
        "run": run,
        "judge": "",
        "created": datetime.now().isoformat(timespec="seconds"),
        "control": None,
        "items": items,
        "metrics": metrics,
    }
    sheet["totals"] = totals(sheet)
    return sheet


def item_score(entry: dict[str, Any]) -> float | None:
    if entry.get("judge") is not None:
        return float(entry["judge"])
    if entry.get("auto") is not None:
        return float(entry["auto"])
    return None


def totals(sheet: dict[str, Any]) -> dict[str, Any]:
    milestone = sheet["milestone"]
    base = bonus = 0.0
    pending = []
    for item in items_for(milestone):
        score = item_score(sheet["items"][item.key])
        if score is None:
            pending.append(item.key)
            continue
        if item.bonus:
            bonus += score
        else:
            base += score
    total = base + bonus
    capped = milestone == 3 and sheet.get("control") == "line_of_sight"
    if capped:
        total *= LINE_OF_SIGHT_FACTOR
    return {
        "base": round(base, 2),
        "bonus": round(bonus, 2),
        "total": round(total, 2),
        "out_of": BASE_TOTAL[milestone],
        "line_of_sight_cap": capped,
        "pending": pending,
    }


def save_sheet(path: Path, sheet: dict[str, Any]) -> None:
    sheet["totals"] = totals(sheet)
    text = yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True, width=110)
    path.write_text(text, encoding="utf-8")


def load_sheet(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / SHEET_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No scoresheet at {path}")
    sheet = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(sheet, dict) or sheet.get("milestone") not in RUBRICS:
        raise ValueError(f"{path} is not a scoresheet")
    return sheet


def format_sheet(sheet: dict[str, Any]) -> str:
    milestone = sheet["milestone"]
    lines = [
        f"MIE444 Milestone {milestone}  team {sheet.get('team') or '?'}  run {sheet.get('run') or '?'}"
        + (f"  judge {sheet['judge']}" if sheet.get("judge") else "")
    ]
    section = None
    for item in items_for(milestone):
        entry = sheet["items"][item.key]
        if item.section != section:
            section = item.section
            lines.append(f"  {section}")
        score = item_score(entry)
        shown = "  --" if score is None else f"{score:4g}"
        src = (
            "judge" if entry.get("judge") is not None else "auto" if entry.get("auto") is not None else "todo"
        )
        out_of = f"/{item.max:g}" if item.max else "   "
        lines.append(f"    {entry['label'][:52]:<52} {shown}{out_of:<4} {src:<5} {entry.get('note') or ''}")
    t = totals(sheet)
    cap = f" x{LINE_OF_SIGHT_FACTOR} line of sight" if t["line_of_sight_cap"] else ""
    lines.append(f"  Total {t['total']:g} / {t['out_of']} (bonus {t['bonus']:g}){cap}")
    if t["pending"]:
        lines.append(f"  Still to mark: {', '.join(t['pending'])}")
    return "\n".join(lines)


def rank_speed(sheets: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    """Give each finished run speed points by rank, fastest first.

    Milestone 2 runs finished inside 5 minutes keep at least 2 of the 4 points, per the rubric.
    """
    sheets = list(sheets)
    eligible = [
        s for s in sheets if s["metrics"]["state"] == "finished" and s["metrics"]["speed_steps_per_min"] > 0
    ]
    eligible.sort(key=lambda s: s["metrics"]["speed_steps_per_min"], reverse=True)
    ranked = []
    n = len(eligible)
    for i, sheet in enumerate(eligible):
        milestone = sheet["milestone"]
        top = SPEED_MAX[milestone]
        share = 1.0 if n == 1 else (n - 1 - i) / (n - 1)
        floor = 2.0 if milestone == 2 and sheet["metrics"]["within_time_limit"] else 0.0
        points = floor + (top - floor) * share
        ranked.append((sheet, round(points * 2) / 2))
    for sheet in sheets:
        if sheet not in eligible:
            ranked.append((sheet, 0.0))
    return ranked
