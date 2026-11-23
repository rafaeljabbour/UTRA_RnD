"""Write and print the autograder result of a run."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

RESULT_NAME = "result.json"
SUMMARY_NAME = "results.csv"
SUMMARY_COLUMNS = (
    "time",
    "team",
    "run",
    "outcome",
    "finish_reason",
    "elapsed_s",
    "started_in_start_zone",
    "ended_in_end_zone",
    "end_zone_share",
    "start_cell",
    "end_cell",
    "distance_ft",
    "steps",
    "minor_collisions",
    "major_collisions",
    "collision_tracking",
    "calibrated",
    "folder",
)


def build_result(team: str, run: str, metrics: dict[str, Any], calibrated: bool) -> dict[str, Any]:
    return {
        "team": team,
        "run": run,
        "time": datetime.now().isoformat(timespec="seconds"),
        "calibrated": calibrated,
        **metrics,
    }


def save_result(run_dir: Path, result: dict[str, Any]) -> Path:
    """Write result.json in the run folder and add a row to results.csv next to the run folders."""
    path = run_dir / RESULT_NAME
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = run_dir.parent / SUMMARY_NAME
    new = not summary.is_file()
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(SUMMARY_COLUMNS)
        row = {**result, "folder": run_dir.name}
        writer.writerow([_cell(row.get(col)) for col in SUMMARY_COLUMNS])
    return path


def format_result(result: dict[str, Any]) -> str:
    def yes(flag: bool) -> str:
        return "yes" if flag else "no"

    collisions = (
        f"{result['minor_collisions']} minor, {result['major_collisions']} major"
        if result["collision_tracking"] or result["collisions"]
        else "not tracked"
    )
    lines = [
        f"Team {result['team'] or '?'}  run {result['run'] or '?'}: {result['outcome']}",
        f"  finished      {result['finish_reason']} after {result['elapsed_s']:.1f} s",
        f"  start zone    {yes(result['started_in_start_zone'])} "
        f"({result['start_zone_share']:.0%} inside, cell {result['start_cell']})",
        f"  end zone      {yes(result['ended_in_end_zone'])} "
        f"({result['end_zone_share']:.0%} inside, cell {result['end_cell']})",
        f"  travelled     {result['distance_ft']:.1f} ft, {result['steps']} one foot steps",
        f"  wall contact  {collisions}",
    ]
    if not result.get("calibrated", True):
        lines.append("  warning       camera not calibrated, whole frame taken as the maze")
    return "\n".join(lines)


def _cell(value: Any) -> Any:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return "" if value is None else value
