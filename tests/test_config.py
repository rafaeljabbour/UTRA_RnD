from pathlib import Path

import pytest

from utra_tracker.config import Config, ConfigError, config_from_dict, load_config, with_overrides

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_yaml_loads():
    cfg = load_config(REPO / "config" / "default.yaml")
    assert cfg.source == 0
    assert cfg.model.weights == "models/robot_detector.pt"
    assert cfg.model.classes and all(isinstance(c, str) for c in cfg.model.classes)
    assert cfg.detect.device == "auto"
    assert cfg.tracker == "bytetrack.yaml"
    assert cfg.output.dir == "runs"
    assert cfg.output.save_video is False


def test_partial_yaml_uses_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "source: video.mp4\ndetect:\n  conf: 0.4\n"))
    assert cfg.source == "video.mp4"
    assert cfg.detect.conf == 0.4
    assert cfg.detect.imgsz == Config().detect.imgsz


def test_empty_yaml_is_default(tmp_path):
    assert load_config(write(tmp_path, "")) == Config()


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_bad_yaml_raises(tmp_path):
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write(tmp_path, "source: [unclosed\n"))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"bogus": 1}, "Unknown key"),
        ({"detect": {"confidence": 0.5}}, "Unknown key in 'detect'"),
        ({"detect": "fast"}, "must be a mapping"),
        ({"detect": {"conf": 1.5}}, "detect.conf"),
        ({"detect": {"conf": 0}}, "detect.conf"),
        ({"detect": {"imgsz": 500}}, "imgsz"),
        ({"detect": {"device": "gpu"}}, "device"),
        ({"model": {"classes": []}}, "classes"),
        ({"model": {"classes": ["ok", ""]}}, "classes"),
        ({"source": -1}, "source"),
        ({"source": True}, "source"),
        ({"tracker": "bytetrack"}, "tracker"),
        ({"output": {"save_video": "yes"}}, "save_video"),
    ],
)
def test_invalid_values(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_overrides_apply_and_validate():
    cfg = with_overrides(Config(), source="clip.mp4", conf=0.5, display=False, save_video=True)
    assert cfg.source == "clip.mp4"
    assert cfg.detect.conf == 0.5
    assert cfg.display is False
    assert cfg.output.save_video is True


def test_overrides_none_keeps_values():
    base = Config()
    assert with_overrides(base) == base


def test_override_bad_conf_raises():
    with pytest.raises(ConfigError):
        with_overrides(Config(), conf=2.0)


def test_to_dict_is_json_friendly():
    data = Config().to_dict()
    assert isinstance(data["model"]["classes"], list)
    assert data["output"]["save_video"] is False
