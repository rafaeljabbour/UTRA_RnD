from pathlib import Path

import pytest

from utra_tracker.cli import build_parser, main


def test_run_flags_parse():
    args = build_parser().parse_args(
        ["run", "--source", "1", "--conf", "0.4", "--no-show", "--save-video", "--max-frames", "10"]
    )
    assert args.source == "1"
    assert args.conf == 0.4
    assert args.no_show and args.save_video
    assert args.max_frames == 10


def test_max_frames_must_be_positive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--max-frames", "0"])


def test_run_with_bad_config_exits_cleanly(tmp_path: Path, capsys):
    assert main(["run", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "Config file not found" in capsys.readouterr().err
