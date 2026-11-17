"""Typed configuration loaded from config.yaml.

Every field has a default, so a config file only needs the values you want to
change. Unknown keys raise an error so typos don't silently do nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"  # "" leaves the driver default
    backend: str = "auto"  # auto | any | dshow | msmf | v4l2
    calibration_file: str = "calib.yaml"
    # Extra VideoCapture properties, by CAP_PROP_* suffix, e.g. {AUTOFOCUS: 0, FOCUS: 0}.
    properties: dict[str, float] = dc_field(default_factory=dict)


@dataclass
class FieldConfig:
    # A WPILib AprilTagField name (e.g. k2026RebuiltWelded) or a path to a layout JSON.
    layout: str = "k2026RebuiltWelded"
    origin: str = "blue"  # blue | red
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.1651  # 6.5 in black-square edge
    bounds_margin_m: float = 0.0  # >0 keeps points slightly outside the walls, <0 shrinks
    ignore_tag_ids: list[int] = dc_field(default_factory=list)


@dataclass
class AprilTagConfig:
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    num_threads: int = 4
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    # WPILib defaults this to 300, which throws away every tag smaller than ~75 px.
    # Tags seen from the driver station are much smaller than that.
    min_cluster_pixels: int = 20
    bits_corrected: int = 1
    max_hamming: int = 0
    min_decision_margin: float = 25.0


@dataclass
class LocalizationConfig:
    interval_s: float = 6.0  # time between re-solves once a pose is held
    retry_interval_s: float = 0.5  # while no pose yet, or confirming a possible camera move
    min_tags: int = 2  # single-tag solves at driver-station range are too noisy to trust alone
    max_reprojection_error_px: float = 2.0
    max_single_tag_ambiguity: float = 0.2  # best/alternative error ratio; single-tag only
    max_jump_m: float = 0.25
    max_jump_deg: float = 2.0
    smoothing_alpha: float = 0.3  # 1 = snap to new solve, 0 = never move
    # Accept a "jump" after this many consecutive, mutually consistent outliers
    # (camera really was bumped). 0 disables.
    jump_confirmations: int = 3
    max_pose_age_s: float = 0.0  # 0 = pose never expires


@dataclass
class DetectorConfig:
    model_path: str = "models/best.pt"
    device: str = "cuda:0"
    half: bool = True
    imgsz: int = 640
    confidence: float = 0.5
    iou: float = 0.5
    classes: list[Any] = dc_field(default_factory=list)  # names or ids; empty = all


@dataclass
class ProjectionConfig:
    floor_z_m: float = 0.0
    ground_offset_m: float = 0.0


@dataclass
class TrackerConfig:
    max_match_distance_m: float = 0.5
    max_missed_frames: int = 10
    min_hits: int = 3
    smoothing_alpha: float = 0.5


@dataclass
class NetworkTablesConfig:
    enabled: bool = True
    table: str = "FieldMapper"
    server: str = ""  # explicit host/IP; empty = use team number + Driver Station
    client_name: str = "FieldMapper"


@dataclass
class DebugConfig:
    show_window: bool = True
    camera_window_scale: float = 1.0
    field_px_per_m: float = 60.0


@dataclass
class CalibrationConfig:
    board_cols: int = 9  # inner corners
    board_rows: int = 6
    square_size_m: float = 0.025
    min_frames: int = 15


@dataclass
class Config:
    team_number: int = 0
    camera: CameraConfig = dc_field(default_factory=CameraConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    apriltag: AprilTagConfig = dc_field(default_factory=AprilTagConfig)
    localization: LocalizationConfig = dc_field(default_factory=LocalizationConfig)
    detector: DetectorConfig = dc_field(default_factory=DetectorConfig)
    projection: ProjectionConfig = dc_field(default_factory=ProjectionConfig)
    tracker: TrackerConfig = dc_field(default_factory=TrackerConfig)
    networktables: NetworkTablesConfig = dc_field(default_factory=NetworkTablesConfig)
    debug: DebugConfig = dc_field(default_factory=DebugConfig)
    calibration: CalibrationConfig = dc_field(default_factory=CalibrationConfig)

    base_dir: Path = dc_field(default_factory=Path.cwd, repr=False)

    def resolve_path(self, p: str) -> Path:
        """Relative paths in the config are relative to the config file."""
        path = Path(p).expanduser()
        return path if path.is_absolute() else (self.base_dir / path)


def _build(cls, data: dict[str, Any], where: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"config section '{where}' must be a mapping")
    known = {f.name: f for f in dataclasses.fields(cls) if f.name != "base_dir"}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config key(s) in '{where}': {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if known[name].default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{where}.{name}" if where else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = _build(Config, data, "")
    cfg.base_dir = path.resolve().parent
    return cfg
