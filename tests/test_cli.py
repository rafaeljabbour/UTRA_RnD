import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_check_reports_broken_cuda(tmp_path: Path, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "Quadro M4000")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.1", cuda=cuda, zeros=broken))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(__version__="8.4.0"))
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    config = tmp_path / "cfg.yaml"
    config.write_text(f"model:\n  weights: {weights.as_posix()}\n", encoding="utf-8")
    assert main(["check", "--config", str(config)]) == 1
    assert "cuda works   no, run install.py --reinstall-torch" in capsys.readouterr().out
