"""Typed, validated configuration loaded from YAML."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from utra_tracker.maze import DEFAULT_LAYOUT, Maze, MazeError, ZoneSpec, validate_layout

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda|cuda:\d+|\d+)$")
FOOTPRINTS = ("mask", "box")


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
class ArenaConfig:
    # One string per row of 1 ft cells: . open, # block, L loading zone, B drop off.
    layout: tuple[str, ...] = DEFAULT_LAYOUT
    cell_ft: float = 1.0
    # Written by `utra-track calibrate`. Without it the whole frame is taken as the arena.
    calibration: str = "config/arena_calibration.json"
    # Set both heights to correct for the robot top being seen above the floor. null disables it.
    camera_height_ft: float | None = None
    robot_height_ft: float = 0.5


@dataclass(frozen=True)
class GradingConfig:
    # Autograder: follow the robot from the start zone to the end zone. false shows the plain tracker.
    enabled: bool = True
    team: str = ""
    run: str = ""
    # Zones are layout letters such as "L" or "B", or a list of [col, row] cells.
    start_zone: ZoneSpec = "L"
    end_zone: ZoneSpec = "B"
    # Share of the robot outline that must be inside a zone to count as in it.
    zone_share: float = 0.5
    # The clock starts once the robot moves this far from where it was placed.
    start_move_ft: float = 0.25
    # The run ends once the robot has been still inside the end zone this long.
    settle_s: float = 2.0
    # The run ends as a fail after this long. null means no limit.
    time_limit_s: float | None = 480.0
    # Wall contact tracking, reported only. Toggle live with the c key.
    collisions: bool = True
    footprint: str = "mask"
    contact_margin_ft: float = 0.04
    major_depth_ft: float = 0.15
    major_duration_s: float = 1.0
    contact_frames: int = 3
    release_frames: int = 6


@dataclass(frozen=True)
class Config:
    source: int | str = 0
    tracker: str = "bytetrack.yaml"
    display: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    grading: GradingConfig = field(default_factory=GradingConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"]["classes"] = list(self.model.classes)
        data["arena"]["layout"] = list(self.arena.layout)
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
        "arena": ArenaConfig,
        "grading": GradingConfig,
    }
    for name, cls in sections.items():
        top[name] = cls(**_known_keys(cls, top.get(name), name))
    if isinstance(top["model"].classes, list):
        top["model"] = replace(top["model"], classes=tuple(top["model"].classes))
    if isinstance(top["arena"].layout, list):
        top["arena"] = replace(top["arena"], layout=tuple(top["arena"].layout))
    for key in ("team", "run"):
        value = getattr(top["grading"], key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            top["grading"] = replace(top["grading"], **{key: str(value)})
    for key in ("start_zone", "end_zone"):
        value = getattr(top["grading"], key)
        if isinstance(value, list):
            cells = tuple(tuple(c) if isinstance(c, list) else c for c in value)
            top["grading"] = replace(top["grading"], **{key: cells})
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
    grading: bool | None = None,
    team: str | None = None,
    run: str | None = None,
    collisions: bool | None = None,
    time_limit_s: float | None = None,
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
    changes = {
        "enabled": grading,
        "team": team,
        "run": run,
        "collisions": collisions,
        "time_limit_s": time_limit_s,
    }
    changes = {k: v for k, v in changes.items() if v is not None}
    if changes:
        cfg = replace(cfg, grading=replace(cfg.grading, **changes))
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
    _validate_arena(cfg.arena)
    _validate_grading(cfg.grading, cfg.arena)


def _validate_arena(arena: ArenaConfig) -> None:
    try:
        validate_layout(arena.layout)
    except MazeError as exc:
        raise ConfigError(str(exc)) from exc
    _require(_is_positive(arena.cell_ft), "arena.cell_ft must be a positive number")
    _require(_is_text(arena.calibration), "arena.calibration must be a path")
    _require(_is_positive(arena.robot_height_ft), "arena.robot_height_ft must be a positive number")
    cam = arena.camera_height_ft
    _require(
        cam is None or (_is_positive(cam) and cam > arena.robot_height_ft),
        "arena.camera_height_ft must be null or higher than arena.robot_height_ft",
    )


def _validate_grading(g: GradingConfig, arena: ArenaConfig) -> None:
    _require(isinstance(g.enabled, bool), "grading.enabled must be true or false")
    _require(isinstance(g.team, str) and isinstance(g.run, str), "grading.team and grading.run must be text")
    maze = Maze(arena.layout, arena.cell_ft)
    for key in ("start_zone", "end_zone"):
        try:
            maze.zone_cells(getattr(g, key))
        except MazeError as exc:
            raise ConfigError(f"grading.{key}: {exc}") from exc
    _require(_is_number(g.zone_share) and 0 < g.zone_share <= 1, "grading.zone_share must be in 0 to 1")
    _require(_is_positive(g.settle_s), "grading.settle_s must be a positive number")
    _require(isinstance(g.collisions, bool), "grading.collisions must be true or false")
    _require(g.footprint in FOOTPRINTS, "grading.footprint must be mask or box")
    _require(
        _is_number(g.contact_margin_ft) and g.contact_margin_ft >= 0, "grading.contact_margin_ft must be >= 0"
    )
    _require(_is_positive(g.major_depth_ft), "grading.major_depth_ft must be a positive number")
    _require(_is_positive(g.major_duration_s), "grading.major_duration_s must be a positive number")
    for key in ("contact_frames", "release_frames"):
        value = getattr(g, key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"grading.{key} must be >= 1",
        )
    _require(_is_positive(g.start_move_ft), "grading.start_move_ft must be a positive number")
    _require(
        g.time_limit_s is None or _is_positive(g.time_limit_s), "grading.time_limit_s must be null or > 0"
    )


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive(value: Any) -> bool:
    return _is_number(value) and value > 0
