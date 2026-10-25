"""Command line entry point for utra-track."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from utra_tracker import __version__
from utra_tracker.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, with_overrides
from utra_tracker.model import ModelError
from utra_tracker.sources import SourceError


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
        return cmd_check(args)
    except (ConfigError, ModelError, SourceError) as exc:
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
    )
    run(cfg, max_frames=args.max_frames)
    return 0


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
