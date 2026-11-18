"""Runs the pipeline on a background thread and exposes state and commands to the web UI.

Designed to start in any state: with no camera, no calibration or no model it
keeps running, reports what is missing, and picks things up as soon as they are
fixed from the browser.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calib_core import CalibrationSession
from .calibration import CameraIntrinsics, load_intrinsics
from .capture import CameraSource, Recorder, VideoSource, probe_cameras
from .config import Config, deep_merge, load_config, save_overrides, to_dict, validate_override
from .field import FieldLayout
from .overlay import Overlay, banner
from .pipeline import FieldMapper
from .publisher import NTPublisher, NullPublisher

log = logging.getLogger("field_mapper.service")

MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}
RESOLUTIONS = [[640, 480], [1280, 720], [1600, 900], [1920, 1080], [2560, 1440], [3840, 2160]]


class LogBuffer(logging.Handler):
    """Keeps recent log records for the web UI."""

    def __init__(self, capacity: int = 400) -> None:
        super().__init__(logging.INFO)
        self._records: deque[dict] = deque(maxlen=capacity)
        self._seq = 0
        self._lock2 = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        with self._lock2:
            self._seq += 1
            self._records.append(
                {"seq": self._seq, "t": record.created, "level": record.levelname, "name": record.name, "msg": msg}
            )

    def since(self, seq: int, limit: int = 200) -> list[dict]:
        with self._lock2:
            return [r for r in self._records if r["seq"] > seq][-limit:]


@dataclass
class Snapshot:
    image: np.ndarray
    index: int
    timestamp: float
    wall: float
    pose: object = None
    pose_valid: bool = False
    loc: object = None
    detections: list = field(default_factory=list)
    det_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    det_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    tracks: list = field(default_factory=list)  # (id, x, y, class_id, conf, misses)
    timing: dict = field(default_factory=dict)
    next_solve_in: float | None = None


class MapperService:
    def __init__(
        self,
        config_path: str | Path,
        *,
        replay: str | None = None,
        replay_speed: float | None = None,
        loop: bool | None = None,
        nt_enabled: bool = True,
        calib_override: str | None = None,
        model_override: str | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        # Command-line overrides last for the session unless the user changes the same setting in the UI.
        self._cli: dict = {}
        if replay:
            self._cli["source"] = {"type": "video", "video_path": str(Path(replay).resolve())}
            if replay_speed is not None:
                self._cli["source"]["speed"] = replay_speed
            if loop is not None:
                self._cli["source"]["loop"] = loop
        if calib_override:
            self._cli.setdefault("camera", {})["calibration_file"] = str(Path(calib_override).resolve())
        if model_override:
            self._cli.setdefault("detector", {})["model_path"] = str(Path(model_override).resolve())
        self._nt_allowed = nt_enabled
        self.cfg = self._load_cfg()
        self.base_dir = self.cfg.base_dir
        self.models_dir = self.base_dir / "models"
        self.recordings_dir = self.base_dir / "recordings"

        self.logs = LogBuffer()
        logging.getLogger().addHandler(self.logs)

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._rebuild = threading.Event()
        self._rebuild.set()
        self._applied_cfg: Config | None = None
        self._thread: threading.Thread | None = None

        self.publisher = NullPublisher()
        self.field: FieldLayout | None = None
        self.field_error: str | None = None
        self.field_version = 0
        self.intr_base: CameraIntrinsics | None = None
        self.intr_error: str | None = None
        self.mapper: FieldMapper | None = None
        self.overlay: Overlay | None = None
        self.source = None
        self.source_error: str | None = None
        self.paused = False
        self.ended = False
        self.recorder: Recorder | None = None
        self.calib: CalibrationSession | None = None

        self.detector = None
        self.detector_error: str | None = None
        self.detector_loading = False
        self._detector_key = None
        self._detector_gen = 0
        self._loaded_detector = None

        self.frame_size: tuple[int, int] | None = None
        self.fps = 0.0
        self._last_frame_wall = 0.0
        self._frame_counter = 0
        self._snap: Snapshot | None = None
        self._jpeg_cache: tuple | None = None
        self._footprint_cache: tuple | None = None
        self._placeholder_cache: tuple | None = None
        self._boot = time.time()
        self._model_load_not_before = 0.0

    # ------------------------------------------------------------------ config
    def _load_cfg(self) -> Config:
        cfg = load_config(self.config_path)
        if self._cli:
            from .config import _build  # same validation as files

            merged = deep_merge(to_dict(cfg), self._cli)
            base = cfg.base_dir
            cfg = _build(Config, merged, "")
            cfg.base_dir = base
        return cfg

    # --------------------------------------------------------------- lifecycle
    def start(self, defer_model_load_s: float = 0.0) -> None:
        """Start the pipeline thread. ``defer_model_load_s`` lets the web server come up
        before the (slow, GIL-heavy) PyTorch import starts."""
        self._model_load_not_before = time.monotonic() + defer_model_load_s
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            if self.recorder:
                self.recorder.close()
                self.recorder = None
            if self.source is not None:
                self.source.close()
                self.source = None
        self.publisher.close()
        logging.getLogger().removeHandler(self.logs)

    @property
    def finished(self) -> bool:
        """True when a non-looping replay has ended (used by headless mode to exit)."""
        return self.ended and isinstance(self.source, VideoSource) and not self.cfg.source.loop

    def _run(self) -> None:
        next_open = 0.0
        while not self._stop.is_set():
            if self._rebuild.is_set():
                self._rebuild.clear()
                try:
                    self._apply_config()
                except Exception:
                    log.exception("failed to apply settings")
            if self.source is None:
                if time.monotonic() >= next_open:
                    self._open_source()
                    next_open = time.monotonic() + 2.0
                if self.source is None:
                    self._stop.wait(0.2)
                    continue
            if self.paused or (self.ended and isinstance(self.source, VideoSource)):
                self._stop.wait(0.05)
                continue
            frame = self.source.read(timeout=0.5)
            if frame is None:
                if isinstance(self.source, VideoSource):
                    if not self.ended:
                        log.info("replay finished")
                    self.ended = True
                continue
            try:
                self._process(frame)
            except Exception:
                log.exception("error processing frame")
                self._stop.wait(0.5)

    def _apply_config(self) -> None:
        """(Re)build whatever the new config affects. Runs on the pipeline thread."""
        old, new = self._applied_cfg, self.cfg
        self._applied_cfg = copy.deepcopy(new)

        # NetworkTables
        nt_on = self._nt_allowed and new.networktables.enabled
        if old is None or (
            old.networktables.enabled != new.networktables.enabled
            or old.networktables.table != new.networktables.table
            or old.networktables.client_name != new.networktables.client_name
        ):
            self.publisher.close()
            if nt_on:
                nt = new.networktables
                self.publisher = NTPublisher(new.team_number, nt.table, nt.server, nt.client_name)
            else:
                self.publisher = NullPublisher()
                log.info("NetworkTables publishing disabled")
        elif (self.publisher.team, self.publisher.server) != (new.team_number, new.networktables.server):
            self.publisher.configure(new.team_number, new.networktables.server)

        # Frame source
        if old is None or old.camera != new.camera or old.source != new.source:
            with self._lock:
                self._close_recorder()
                if self.source is not None:
                    self.source.close()
                self.source = None
                self.source_error = None
                self.ended = False
                self.paused = False
                self._snap = None

        # Field layout
        if old is None or old.field != new.field or self.field is None:
            try:
                layout = new.field.layout
                p = new.resolve_path(layout)
                self.field = FieldLayout.load(
                    str(p) if p.exists() else layout, new.field.tag_size_m, new.field.origin, new.field.ignore_tag_ids
                )
                self.field_error = None
                self.field_version += 1
                log.info("field layout %s: %.2f x %.2f m, %d tags", layout, self.field.length, self.field.width, len(self.field.tags))
            except Exception as e:
                self.field, self.field_error = None, str(e)
                log.error("field layout: %s", e)

        self._reload_intrinsics()
        self._start_detector_load()
        with self._lock:
            self.mapper = None  # rebuilt on the next frame with the new settings
            self.overlay = None

    def _reload_intrinsics(self) -> None:
        path = self.cfg.resolve_path(self.cfg.camera.calibration_file)
        try:
            self.intr_base = load_intrinsics(path)
            self.intr_error = None
            log.info("calibration %s: %dx%d, RMS %s px", path.name, self.intr_base.width, self.intr_base.height,
                     f"{self.intr_base.rms_px:.2f}" if self.intr_base.rms_px is not None else "?")
        except FileNotFoundError:
            self.intr_base = None
            self.intr_error = "not calibrated"
        except Exception as e:
            self.intr_base, self.intr_error = None, f"could not read {path.name}: {e}"
            log.error("calibration: %s", self.intr_error)

    def _start_detector_load(self) -> None:
        d = self.cfg.detector
        path = self.cfg.resolve_path(d.model_path)
        # The file's size/mtime are part of the key: uploading a model over the same path
        # (e.g. best.pt onto the default models/best.pt) must load it again.
        try:
            st = path.stat()
            file_sig = (st.st_size, st.st_mtime_ns)
        except OSError:
            file_sig = None
        key = (str(path), file_sig, str(d.device), bool(d.half), int(d.imgsz), tuple(str(c) for c in d.classes))
        if key == self._detector_key:
            if self.detector is not None:
                self.detector.set_thresholds(d.confidence, d.iou)
            return
        self._detector_key = key
        self._detector_gen += 1
        gen = self._detector_gen
        self.detector = None
        if self.mapper is not None:
            self.mapper.detector = None
        if not path.exists():
            self.detector_error = f"no model file at {self._rel(path)}"
            self.detector_loading = False
            return
        self.detector_loading = True
        self.detector_error = None
        dcfg = dataclasses.replace(d, classes=list(d.classes))

        def work() -> None:
            wait = self._model_load_not_before - time.monotonic()
            if wait > 0:
                self._stop.wait(wait)
            if gen != self._detector_gen or self._stop.is_set():
                return
            det, err = None, None
            try:
                from .detector import YoloDetector

                det = YoloDetector(dcfg, str(path))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                log.error("could not load model %s: %s", path.name, err)
            with self._lock:
                if gen != self._detector_gen:
                    return  # superseded by a newer settings change
                self._loaded_detector = det
                self.detector_error = err
                self.detector_loading = False

        threading.Thread(target=work, name="model-load", daemon=True).start()

    def _open_source(self) -> None:
        cfg = self.cfg
        try:
            if cfg.source.type == "video":
                if not cfg.source.video_path:
                    raise ValueError("no video file selected")
                src = VideoSource(cfg.resolve_path(cfg.source.video_path), cfg.source.speed, cfg.source.loop)
            else:
                src = CameraSource(cfg.camera)
        except Exception as e:
            msg = str(e)
            if msg != self.source_error:
                log.warning("video source: %s", msg)
            self.source_error = msg
            return
        with self._lock:
            self.source = src
            self.source_error = None
            self.ended = False

    # ---------------------------------------------------------------- per frame
    def _process(self, frame) -> None:
        img = frame.image
        h, w = img.shape[:2]
        self.frame_size = (w, h)
        if self.recorder is not None:
            self.recorder.write(img)

        with self._lock:
            if self._loaded_detector is not None:
                self.detector, self._loaded_detector = self._loaded_detector, None
                if self.mapper is not None:
                    self.mapper.detector = self.detector
            calib = self.calib

        snap = Snapshot(img, frame.index, frame.timestamp, time.monotonic())
        if calib is not None:
            calib.on_frame(img)
            self.publisher.publish([], [], frame.capture_monotonic, [0.0] * 6, False)
        else:
            mapper = self._ensure_mapper(w, h)
            if mapper is not None:
                mapper.step(img, frame.timestamp, frame.capture_monotonic)
                loc = mapper.localizer
                snap.pose = loc.pose
                snap.pose_valid = loc.pose_valid(frame.timestamp)
                snap.loc = loc.last_result
                snap.detections = mapper.detections
                snap.det_xy, snap.det_ok = mapper.det_xy, mapper.det_ok
                snap.tracks = [(t.track_id, t.x, t.y, t.class_id, t.confidence, t.misses) for t in mapper.tracks]
                snap.timing = dict(mapper.timing)
                nt = loc._next_time
                snap.next_solve_in = None if nt is None else max(0.0, nt - frame.timestamp)
            else:
                self.publisher.publish([], [], frame.capture_monotonic, [0.0] * 6, False)

        now = time.monotonic()
        if self._last_frame_wall:
            dt = now - self._last_frame_wall
            if dt > 0:
                self.fps = 1.0 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
        self._last_frame_wall = now
        with self._lock:
            self._snap = snap

    def _ensure_mapper(self, w: int, h: int) -> FieldMapper | None:
        if self.mapper is not None and (self.mapper.intr.width, self.mapper.intr.height) == (w, h):
            return self.mapper
        if self.intr_base is None or self.field is None:
            return None
        try:
            intr = self.intr_base.scaled_to(w, h)
        except ValueError as e:
            if self.intr_error != str(e):
                log.error("%s", e)
            self.intr_error = str(e)
            return None
        with self._lock:
            self.mapper = FieldMapper(self.cfg, intr, self.field, self.detector, self.publisher)
            self.overlay = Overlay(self.field, intr.camera_matrix, intr.dist_coeffs, (w, h))
        return self.mapper

    # ---------------------------------------------------------------- commands
    def apply_settings(self, values: dict) -> None:
        """Validate, persist to settings.yaml and apply. Raises ValueError with a readable message."""
        merged = validate_override(self.config_path, values)
        save_overrides(self.config_path, merged)
        self._drop_cli(values)
        self.cfg = self._load_cfg()
        log.info("settings saved")
        self._rebuild.set()

    def _drop_cli(self, values: dict, cli: dict | None = None) -> None:
        cli = self._cli if cli is None else cli
        for k, v in values.items():
            if k in cli:
                if isinstance(v, dict) and isinstance(cli[k], dict):
                    self._drop_cli(v, cli[k])
                    if not cli[k]:
                        del cli[k]
                else:
                    del cli[k]

    def set_confidence(self, value: float) -> None:
        value = float(np.clip(value, 0.01, 0.99))
        self.cfg.detector.confidence = value
        if self.detector is not None:
            self.detector.set_thresholds(value)
        merged = validate_override(self.config_path, {"detector": {"confidence": value}})
        save_overrides(self.config_path, merged)

    def relocalize(self) -> None:
        if self.mapper is not None:
            self.mapper.localizer.force()
            log.info("re-localising now")

    def reset_tracker(self) -> None:
        if self.mapper is not None:
            self.mapper.tracker.reset()
            log.info("tracker reset")

    def set_paused(self, paused: bool) -> None:
        if isinstance(self.source, VideoSource):
            self.paused = bool(paused)
            if not self.paused:
                self.source.resync()

    def restart_replay(self) -> None:
        if isinstance(self.source, VideoSource):
            with self._lock:
                self.source.restart()
                self.ended = False
                self.paused = False
            log.info("replay restarted")

    def start_recording(self) -> str:
        if not isinstance(self.source, CameraSource):
            raise ValueError("recording is only available with a live camera")
        if self.frame_size is None:
            raise ValueError("no frames from the camera yet")
        name = _dt.datetime.now().strftime("match-%Y%m%d-%H%M%S.avi")
        fps = self.source.fps or self.cfg.camera.fps or 30.0
        with self._lock:
            self._close_recorder()
            self.recorder = Recorder(self.recordings_dir / name, round(fps), self.frame_size)
        log.info("recording to recordings/%s", name)
        return name

    def stop_recording(self) -> None:
        with self._lock:
            self._close_recorder()

    def _close_recorder(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            log.info("recording saved: recordings/%s (%d frames)", self.recorder.path.name, self.recorder.frames)
            self.recorder = None

    # calibration
    def start_calibration(self, cols: int, rows: int, square_mm: float, min_views: int | None = None) -> None:
        c = self.cfg.calibration
        min_views = int(min_views or c.min_frames)
        merged = validate_override(self.config_path, {"calibration": {
            "board_cols": int(cols), "board_rows": int(rows), "square_size_m": float(square_mm) / 1000.0, "min_frames": min_views}})
        save_overrides(self.config_path, merged)
        self.cfg = self._load_cfg()
        out = self.cfg.resolve_path(self.cfg.camera.calibration_file)
        with self._lock:
            self.calib = CalibrationSession(cols, rows, float(square_mm) / 1000.0, min_views, out)
        log.info("calibration started (%dx%d inner corners, %.1f mm squares)", cols, rows, square_mm)

    def stop_calibration(self) -> None:
        with self._lock:
            session, self.calib = self.calib, None
        if session is not None:
            self._reload_intrinsics()
            with self._lock:
                self.mapper = None
                self.overlay = None
            log.info("calibration closed")

    # files
    def list_models(self) -> list[str]:
        out = []
        if self.models_dir.exists():
            out = sorted(self._rel(p) for p in self.models_dir.iterdir() if p.suffix.lower() in MODEL_SUFFIXES)
        current = self.cfg.detector.model_path
        if current and current not in out and self.cfg.resolve_path(current).exists():
            out.append(current)
        return out

    def list_recordings(self) -> list[str]:
        out = []
        if self.recordings_dir.exists():
            files = [p for p in self.recordings_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
            out = [self._rel(p) for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]
        current = self.cfg.source.video_path
        if current and current not in out and self.cfg.resolve_path(current).exists():
            out.insert(0, current)
        return out

    def save_model(self, filename: str, data: bytes) -> str:
        name = Path(filename).name
        if Path(name).suffix.lower() not in MODEL_SUFFIXES:
            raise ValueError(f"model must be one of {', '.join(sorted(MODEL_SUFFIXES))}")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / name).write_bytes(data)
        log.info("model uploaded: models/%s (%.1f MB)", name, len(data) / 1e6)
        return f"models/{name}"

    def _rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.base_dir).as_posix()
        except ValueError:
            return str(p)

    def probe_cameras(self) -> list[dict]:
        active = self.cfg.camera.index if isinstance(self.source, CameraSource) else None
        found = probe_cameras(self.cfg.camera.backend, skip={active} if active is not None else None)
        if active is not None:
            with self._lock:
                snap = self._snap
            jpeg = None
            if snap is not None:
                h, w = snap.image.shape[:2]
                thumb = cv2.resize(snap.image, (320, int(h * 320 / w)), interpolation=cv2.INTER_AREA)
                jpeg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            size = self.frame_size or (0, 0)
            found.append({"index": active, "width": size[0], "height": size[1], "jpeg": jpeg, "active": True})
        return sorted(found, key=lambda c: c["index"])

    # ------------------------------------------------------------------ state
    def settings_view(self) -> dict:
        import robotpy_apriltag as ra

        cfg = self.cfg
        layouts = sorted((n for n in ra.AprilTagField.__members__ if n.startswith("k20")), reverse=True)
        return {
            "values": {
                "team_number": cfg.team_number,
                "networktables": {"enabled": cfg.networktables.enabled, "server": cfg.networktables.server},
                "source": dataclasses.asdict(cfg.source),
                "camera": {"index": cfg.camera.index, "width": cfg.camera.width, "height": cfg.camera.height, "fps": cfg.camera.fps},
                "field": {"layout": cfg.field.layout, "origin": cfg.field.origin},
                "detector": {"model_path": cfg.detector.model_path, "confidence": cfg.detector.confidence,
                             "device": str(cfg.detector.device), "classes": [str(c) for c in cfg.detector.classes]},
                "localization": {"interval_s": cfg.localization.interval_s, "min_tags": cfg.localization.min_tags},
                "projection": {"ground_offset_m": cfg.projection.ground_offset_m},
                "web": {"host": cfg.web.host},
                "calibration": {"board_cols": cfg.calibration.board_cols, "board_rows": cfg.calibration.board_rows,
                                "square_size_m": cfg.calibration.square_size_m, "min_frames": cfg.calibration.min_frames},
            },
            "options": {
                "layouts": layouts,
                "models": self.list_models(),
                "recordings": self.list_recordings(),
                "resolutions": RESOLUTIONS,
                "devices": ["cuda:0", "cpu"],
            },
            "cli_overrides": sorted(self._cli),
        }

    def field_view(self) -> dict:
        f = self.field
        if f is None:
            return {"ok": False, "error": self.field_error, "version": self.field_version}
        tags = []
        for tid, t in sorted(f.tags.items()):
            normal = t.rotation[:, 0]
            tags.append({"id": tid, "x": float(t.position[0]), "y": float(t.position[1]), "z": float(t.position[2]),
                         "yaw": float(np.arctan2(normal[1], normal[0]))})
        return {"ok": True, "length": f.length, "width": f.width, "tags": tags,
                "origin": self.cfg.field.origin, "layout": self.cfg.field.layout, "version": self.field_version}

    def state(self) -> dict:
        with self._lock:
            snap = self._snap
            calib = self.calib
            mapper = self.mapper
            overlay = self.overlay
            recorder = self.recorder
            source = self.source
        cfg = self.cfg
        now = time.monotonic()
        live = snap is not None and now - snap.wall < 2.0

        if isinstance(source, VideoSource):
            total = source.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            pos = source.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0
            src = {"type": "video", "name": Path(source.path).name, "paused": self.paused, "ended": self.ended,
                   "position_s": pos / source.fps, "duration_s": total / source.fps if total else None}
        else:
            src = {"type": "camera", "index": cfg.camera.index}
        cam_status = "ok" if live or (src["type"] == "video" and (self.paused or self.ended) and snap is not None) else (
            "error" if self.source_error else "waiting")
        camera = {"status": cam_status, "error": self.source_error, "fps": round(self.fps, 1),
                  "width": self.frame_size[0] if self.frame_size else None,
                  "height": self.frame_size[1] if self.frame_size else None, **src}

        calibration = {"ok": self.intr_base is not None and self.intr_error is None,
                       "error": self.intr_error, "file": cfg.camera.calibration_file}
        if self.intr_base is not None:
            calibration.update(width=self.intr_base.width, height=self.intr_base.height, rms_px=self.intr_base.rms_px)

        model = {"ok": self.detector is not None, "loading": self.detector_loading, "error": self.detector_error,
                 "path": cfg.detector.model_path, "confidence": cfg.detector.confidence}
        if self.detector is not None:
            model.update(device=self.detector.device, half=self.detector.half, classes=list(self.detector.names.values()))

        pose = {"valid": False}
        pieces, raw, footprint, timing, used_tags = [], [], None, {}, []
        if snap is not None and snap.pose is not None:
            p = snap.pose
            x, y, z, roll, pitch, yaw = p.as_list()
            pose = {"valid": snap.pose_valid, "x": x, "y": y, "z": z,
                    "roll_deg": float(np.degrees(roll)), "pitch_deg": float(np.degrees(pitch)), "yaw_deg": float(np.degrees(yaw))}
            if overlay is not None:
                if self._footprint_cache is None or self._footprint_cache[0] is not p:
                    try:
                        fp = overlay.footprint(p).tolist()
                    except Exception:
                        fp = None
                    self._footprint_cache = (p, fp)
                footprint = self._footprint_cache[1]
        if snap is not None:
            if snap.loc is not None:
                loc = snap.loc
                pose["last_solve"] = {
                    "age_s": max(0.0, snap.timestamp - loc.timestamp), "accepted": loc.accepted, "reason": loc.reason,
                    "tags": loc.solution.tag_ids if loc.solution else [], "seen": [o.tag_id for o in loc.observations],
                    "error_px": loc.solution.reprojection_error_px if loc.solution else None,
                }
                if loc.solution and loc.accepted:
                    used_tags = loc.solution.tag_ids
            pose["next_solve_in_s"] = snap.next_solve_in
            names = self.detector.names if self.detector is not None else {}
            pieces = [{"id": tid, "x": tx, "y": ty, "class_id": cid, "class": names.get(cid, str(cid)),
                       "confidence": conf, "seen": misses == 0} for tid, tx, ty, cid, conf, misses in snap.tracks]
            raw = snap.det_xy[snap.det_ok].tolist() if len(snap.det_ok) else []
            timing = snap.timing

        nt = {"enabled": not isinstance(self.publisher, NullPublisher),
              "connected": bool(self.publisher.connected),
              "team": cfg.team_number, "server": cfg.networktables.server}
        return {
            "time": time.time(),
            "boot": self._boot,
            "camera": camera,
            "calibration": calibration,
            "model": model,
            "field": {"ok": self.field is not None, "error": self.field_error, "version": self.field_version,
                      "layout": cfg.field.layout},
            "pose": pose,
            "nt": nt,
            "pieces": pieces,
            "raw": raw,
            "footprint": footprint,
            "used_tags": used_tags,
            "detections": len(snap.detections) if snap is not None else 0,
            "timing": {k: round(v, 1) for k, v in timing.items()},
            "mapping": mapper is not None and calib is None,
            "recording": {"active": recorder is not None, "file": recorder.path.name if recorder else None,
                          "seconds": now - recorder.started if recorder else 0.0},
            "calibrating": calib.state() if calib is not None else {"active": False},
        }

    # ----------------------------------------------------------------- images
    def render_jpeg(self, max_width: int | None = None, quality: int | None = None,
                    show_field: bool = True, show_detections: bool = True) -> bytes:
        max_width = max_width or self.cfg.web.stream_width
        quality = quality or self.cfg.web.jpeg_quality
        with self._lock:
            snap, calib, overlay, mapper = self._snap, self.calib, self.overlay, self.mapper
        if snap is None:
            return self._placeholder(self._waiting_message(), max_width, quality)
        key = (snap.index, id(snap), max_width, quality, show_field, show_detections, calib is not None,
               calib.state()["views"] if calib else 0)
        cache = self._jpeg_cache
        if cache is not None and cache[0] == key:
            return cache[1]
        if calib is not None:
            img = calib.render(snap.image)
        elif overlay is not None and mapper is not None:
            img = overlay.render(snap.image, snap.timestamp, snap.pose, snap.pose_valid, snap.loc,
                                 snap.detections, snap.det_xy, snap.det_ok, show_field, show_detections)
        elif self.intr_base is None or self.intr_error:
            msg = "Camera not calibrated - open the Calibrate tab" if self.intr_base is None else "Calibration doesn't match this resolution - recalibrate"
            img = banner(snap.image, msg)
        else:
            img = snap.image
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        data = buf.tobytes()
        self._jpeg_cache = (key, data)
        return data

    def _waiting_message(self) -> str:
        if self.source_error:
            return f"Camera problem: {self.source_error}"
        return "Starting camera..."

    def _placeholder(self, msg: str, width: int, quality: int) -> bytes:
        key = (msg, width, quality)
        if self._placeholder_cache and self._placeholder_cache[0] == key:
            return self._placeholder_cache[1]
        h = int(width * 9 / 16)
        img = banner(np.full((h, width, 3), 24, np.uint8), msg, (200, 200, 200))
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self._placeholder_cache = (key, data)
        return data

    @property
    def frame_index(self) -> int:
        snap = self._snap
        return -1 if snap is None else snap.index
