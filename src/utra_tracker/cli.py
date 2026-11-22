"""Command line entry point for utra-track."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from utra_tracker import __version__
from utra_tracker.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, with_overrides
from utra_tracker.maze import MazeError
from utra_tracker.model import ModelError, cuda_smoke_test
from utra_tracker.sources import SourceError


class SheetError(RuntimeError):
    """Raised when a scoresheet cannot be read."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="utra-track", description="Overhead maze robot tracker.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file")
        return p

    prep = with_config(sub.add_parser("prepare", help="download base weights and bake prompt classes"))
    prep.add_argument("--force", action="store_true", help="rebuild even if the model file exists")

    run = with_config(sub.add_parser("run", help="detect and track robots"))
    run.add_argument("--source", help="camera index, video file or RTSP or HTTP URL")
    run.add_argument("--conf", type=float, help="detection confidence threshold")
    run.add_argument("--no-show", action="store_true", help="run without a window")
    run.add_argument("--save-video", action="store_true", help="also save annotated video")
    run.add_argument("--max-frames", type=_positive_int, help="stop after this many frames")
    run.add_argument(
        "--milestone", type=int, choices=(0, 2, 3), help="rubric to grade with, 0 turns grading off"
    )
    run.add_argument("--team", help="team number or name for the scoresheet")
    run.add_argument("--run", dest="run_label", help="run or trial number for the scoresheet")
    run.add_argument(
        "--collisions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="track wall contact with the camera (toggle live with c)",
    )
    run.add_argument("--no-marks", action="store_true", help="do not ask for judge marks after the run")

    cal = with_config(sub.add_parser("calibrate", help="click the 4 maze corners so pixels map to feet"))
    cal.add_argument("--source", help="camera index, video, image or stream to calibrate from")
    cal.add_argument("--output", help="where to save, default arena.calibration from the config")

    grade = sub.add_parser("grade", help="enter judge marks for a run and print its total")
    grade.add_argument("run_dir", help="run folder or its scoresheet.yaml")
    grade.add_argument("--show", action="store_true", help="only print the scoresheet")

    board = sub.add_parser("leaderboard", help="rank speed across runs and print totals")
    board.add_argument("--runs", default="runs", help="folder holding the run folders")
    board.add_argument("--milestone", type=int, choices=(2, 3), required=True)
    board.add_argument("--dry-run", action="store_true", help="print without writing speed marks")

    cams = sub.add_parser("cameras", help="list camera indices that open")
    cams.add_argument("--max-index", type=int, default=9)

    with_config(sub.add_parser("check", help="print environment and model status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Runtime must never depend on the network, this also skips Ultralytics update checks.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    try:
        if args.command == "prepare":
            return cmd_prepare(args)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "cameras":
            return cmd_cameras(args)
        if args.command == "calibrate":
            return cmd_calibrate(args)
        if args.command == "grade":
            return cmd_grade(args)
        if args.command == "leaderboard":
            return cmd_leaderboard(args)
        return cmd_check(args)
    except (ConfigError, ModelError, SourceError, MazeError, SheetError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def cmd_prepare(args: argparse.Namespace) -> int:
    # Preparing is the one step that is allowed to download.
    os.environ["YOLO_OFFLINE"] = "0"
    from utra_tracker.model import prepare_model

    prepare_model(load_config(args.config), force=args.force)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from utra_tracker.tracker import run

    cfg = with_overrides(
        load_config(args.config),
        source=args.source,
        conf=args.conf,
        display=False if args.no_show else None,
        save_video=True if args.save_video else None,
        milestone=args.milestone,
        team=args.team,
        run=args.run_label,
        collisions=args.collisions,
    )
    _, sheet_path = run(cfg, max_frames=args.max_frames)
    if sheet_path is not None:
        print(f"Scoresheet: {sheet_path}")
        if not args.no_marks and sys.stdin.isatty() and _ask_yes("Enter judge marks now?"):
            return _mark(sheet_path, input)
        print(f"Enter judge marks later with: utra-track grade {sheet_path.parent}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from utra_tracker.calibrate import run_calibration
    from utra_tracker.maze import Maze

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg.source
    out = Path(args.output or cfg.arena.calibration)
    return 0 if run_calibration(source, Maze(cfg.arena.layout, cfg.arena.cell_ft), out) else 1


def cmd_grade(args: argparse.Namespace) -> int:
    from utra_tracker.rubric import format_sheet

    path = _sheet_path(args.run_dir)
    if args.show:
        print(format_sheet(_load_sheet(path)))
        return 0
    return _mark(path, input)


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from utra_tracker.rubric import load_sheet, rank_speed, save_sheet, totals

    paths = sorted(Path(args.runs).glob("*/scoresheet.yaml"))
    sheets = []
    for path in paths:
        try:
            sheet = load_sheet(path)
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        if sheet["milestone"] == args.milestone:
            sheet["_path"] = path
            sheets.append(sheet)
    if not sheets:
        print(f"No milestone {args.milestone} scoresheets under {args.runs}")
        return 1
    ranked = rank_speed(sheets)
    print(f"Milestone {args.milestone} leaderboard, speed ranked fastest first")
    print(f"  {'team':<10} {'run':<6} {'steps/min':>9} {'time':>7} {'speed':>6} {'total':>7}  folder")
    for sheet, points in ranked:
        path = sheet.pop("_path")
        sheet["items"]["speed"]["auto"] = points
        rank_note = f"ranked {sheet['metrics']['speed_steps_per_min']} steps/min against {len(ranked)} runs"
        sheet["items"]["speed"]["note"] = rank_note
        m = sheet["metrics"]
        t = totals(sheet)
        print(
            f"  {sheet.get('team') or '?':<10} {sheet.get('run') or '?':<6} {m['speed_steps_per_min']:>9} "
            f"{m['elapsed_s']:>6.0f}s {points:>6g} {t['total']:>7g}  {path.parent.name}"
        )
        if not args.dry_run:
            save_sheet(path, sheet)
    if args.dry_run:
        print("Dry run, nothing written.")
    print("Speed marks are a suggestion. A judge mark in a scoresheet still overrides them.")
    return 0


def _mark(path: Path, ask: Callable[[str], str]) -> int:
    """Walk the judge through every item. Enter keeps the shown value."""
    from utra_tracker.rubric import CONTROL_MODES, format_sheet, items_for, save_sheet

    sheet = _load_sheet(path)
    milestone = sheet["milestone"]
    print(f"\nJudge marks for {path}. Enter keeps the value in brackets, '-' clears a judge mark.\n")
    sheet["judge"] = _ask_text(ask, "Judge name", sheet.get("judge") or "")
    sheet["team"] = _ask_text(ask, "Team", sheet.get("team") or "")
    sheet["run"] = _ask_text(ask, "Run", sheet.get("run") or "")
    if milestone == 3:
        modes = "/".join(CONTROL_MODES)
        while True:
            value = _ask_text(
                ask, f"Control ({modes}), line_of_sight caps at 70%", sheet.get("control") or ""
            )
            if value in ("", *CONTROL_MODES):
                sheet["control"] = value or None
                break
            print(f"  pick one of {modes}")
    for item in items_for(milestone):
        entry = sheet["items"][item.key]
        current = entry["judge"] if entry["judge"] is not None else entry["auto"]
        shown = "" if current is None else f"{current:g}"
        note = f"  ({entry['note']})" if entry.get("note") else ""
        limit = f"/{item.max:g}" if item.max else ""
        while True:
            raw = ask(f"{item.label} {limit}{note} [{shown}]: ").strip()
            if raw == "":
                if entry["judge"] is None and entry["auto"] is not None:
                    entry["judge"] = entry["auto"]
                break
            if raw == "-":
                entry["judge"] = None
                break
            try:
                value = float(raw)
            except ValueError:
                print("  enter a number, '-' or nothing")
                continue
            low, high = (-1.0, 0.0) if item.key == "usb_deduction" else (0.0, float(item.max))
            if not low <= value <= high:
                print(f"  must be between {low:g} and {high:g}")
                continue
            entry["judge"] = value
            break
    save_sheet(path, sheet)
    print()
    print(format_sheet(sheet))
    print(f"\nSaved {path}")
    return 0


def _sheet_path(raw: str) -> Path:
    from utra_tracker.rubric import SHEET_NAME

    path = Path(raw)
    return path / SHEET_NAME if path.is_dir() else path


def _load_sheet(path: Path) -> dict:
    from utra_tracker.rubric import load_sheet

    try:
        return load_sheet(path)
    except (OSError, ValueError) as exc:
        raise SheetError(str(exc)) from exc


def _ask_text(ask: Callable[[str], str], label: str, current: str) -> str:
    raw = ask(f"{label} [{current}]: ").strip()
    return current if raw == "" else raw


def _ask_yes(question: str) -> bool:
    try:
        return input(f"{question} [Y/n]: ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def cmd_cameras(args: argparse.Namespace) -> int:
    from utra_tracker.sources import probe_cameras

    print(f"Probing camera indices 0 to {args.max_index}...")
    found = probe_cameras(args.max_index)
    for index, width, height in found:
        print(f"  camera {index}: {width}x{height}")
    if not found:
        print("  no cameras found")
    return 0 if found else 1


def cmd_check(args: argparse.Namespace) -> int:
    print(f"utra-track   {__version__}")
    print(f"python       {platform.python_version()} at {sys.executable}")
    ok = True
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch        {torch.__version__}")
        print(f"cuda         {'available' if cuda else 'not available'}")
        print(f"device       {torch.cuda.get_device_name(0) if cuda else 'cpu'}")
        if cuda:
            works = cuda_smoke_test() is None
            print(f"cuda works   {'yes' if works else 'no, run install.py --reinstall-torch'}")
            ok = ok and works
    except ImportError:
        print("torch        not installed")
        ok = False
    try:
        import ultralytics

        print(f"ultralytics  {ultralytics.__version__}")
    except ImportError:
        print("ultralytics  not installed")
        ok = False
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config       invalid: {exc}")
        return 1
    weights = Path(cfg.model.weights)
    print(f"config       {args.config}")
    print(f"model        {weights} {'found' if weights.is_file() else 'MISSING, run utra-track prepare'}")
    return 0 if ok and weights.is_file() else 1


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value
