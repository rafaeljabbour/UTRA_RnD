"""Typed, validated configuration loaded from YAML."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda|cuda:\d+|\d+)$")


class ConfigError(ValueError):
    """Raised when a config file is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "models/robot_detector.pt"
    base: str = "yoloe-26m-seg.pt"
    classes: tuple[str, ...] = ("white disk robot", "mobile robot", "robot car")


@dataclass(frozen=True)
class DetectConfig:
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "runs"
    save_video: bool = False


@dataclass(frozen=True)
class Config:
    source: int | str = 0
    tracker: str = "bytetrack.yaml"
    display: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"]["classes"] = list(self.model.classes)
        return data


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML file and return a validated Config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    return config_from_dict(data)


def config_from_dict(data: Any) -> Config:
    """Build a Config from a plain mapping and validate it."""
    top = _known_keys(Config, data, "config")
    sections = {
        "model": ModelConfig,
        "detect": DetectConfig,
        "output": OutputConfig,
    }
    for name, cls in sections.items():
        top[name] = cls(**_known_keys(cls, top.get(name), name))
    if isinstance(top["model"].classes, list):
        top["model"] = replace(top["model"], classes=tuple(top["model"].classes))
    cfg = Config(**top)
    validate(cfg)
    return cfg


def with_overrides(
    cfg: Config,
    *,
    source: str | None = None,
    conf: float | None = None,
    display: bool | None = None,
    save_video: bool | None = None,
) -> Config:
    """Return a copy of cfg with CLI overrides applied and validated."""
    if source is not None:
        cfg = replace(cfg, source=source)
    if conf is not None:
        cfg = replace(cfg, detect=replace(cfg.detect, conf=conf))
    if display is not None:
        cfg = replace(cfg, display=display)
    if save_video is not None:
        cfg = replace(cfg, output=replace(cfg.output, save_video=save_video))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Raise ConfigError if any value is out of range or the wrong type."""
    src = cfg.source
    _require(
        (isinstance(src, int) and not isinstance(src, bool) and src >= 0)
        or (isinstance(src, str) and src.strip() != ""),
        "source must be a camera index >= 0, a file path or a stream URL",
    )
    _require(_is_text(cfg.tracker) and cfg.tracker.endswith(".yaml"), "tracker must be a .yaml")
    _require(isinstance(cfg.display, bool), "display must be true or false")
    _require(_is_text(cfg.model.weights), "model.weights must be a path")
    _require(_is_text(cfg.model.base), "model.base must be a model name")
    classes = cfg.model.classes
    _require(
        isinstance(classes, tuple) and len(classes) > 0 and all(_is_text(c) for c in classes),
        "model.classes must be a non empty list of prompt strings",
    )
    _require(_is_unit(cfg.detect.conf), "detect.conf must be a number in 0 to 1")
    _require(_is_unit(cfg.detect.iou), "detect.iou must be a number in 0 to 1")
    imgsz = cfg.detect.imgsz
    _require(
        isinstance(imgsz, int) and not isinstance(imgsz, bool) and imgsz >= 32 and imgsz % 32 == 0,
        "detect.imgsz must be a positive multiple of 32",
    )
    _require(
        isinstance(cfg.detect.device, str) and bool(_DEVICE_RE.match(cfg.detect.device)),
        "detect.device must be auto, cpu, cuda, cuda:N or mps",
    )
    _require(_is_text(cfg.output.dir), "output.dir must be a path")
    _require(isinstance(cfg.output.save_video, bool), "output.save_video must be true or false")


def _known_keys(cls: type, data: Any, name: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown key in '{name}': {', '.join(sorted(map(str, unknown)))}")
    return dict(data)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_unit(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1
