"""Main loop: capture -> localise -> detect -> project -> track -> publish."""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from .calibration import load_intrinsics
from .capture import CameraSource, VideoSource
from .config import Config, load_config
from .field import FieldLayout
from .geometry import project_pixels_to_floor
from .localizer import Localizer
from .publisher import NTPublisher, NullPublisher
from .tracker import Measurement, NearestNeighbourTracker

log = logging.getLogger("field_mapper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FRC driver-station game piece mapper")
    p.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    p.add_argument("--replay", metavar="VIDEO", help="run on a video file instead of the live camera")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time, 0 = as fast as possible)")
    p.add_argument("--loop", action="store_true", help="loop the replay video")
    p.add_argument("--calib", help="override camera.calibration_file")
    p.add_argument("--model", help="override detector.model_path")
    p.add_argument("--no-nt", action="store_true", help="don't publish to NetworkTables")
    window = p.add_mutually_exclusive_group()
    window.add_argument("--debug", dest="window", action="store_true", default=None, help="show the debug windows")
    window.add_argument("--headless", dest="window", action="store_false", help="no debug windows")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


class FieldMapper:
    """Owns all pipeline stages. ``step(frame)`` processes one frame."""

    def __init__(self, cfg: Config, intrinsics, field: FieldLayout, detector, publisher) -> None:
        self.cfg = cfg
        self.intr = intrinsics
        self.field = field
        self.detector = detector
        self.publisher = publisher
        self.localizer = Localizer(
            cfg.localization, cfg.apriltag, field, intrinsics.camera_matrix, intrinsics.dist_coeffs, cfg.field.tag_family
        )
        t = cfg.tracker
        self.tracker = NearestNeighbourTracker(t.max_match_distance_m, t.max_missed_frames, t.min_hits, t.smoothing_alpha)
        # Exposed for the debug view
        self.detections = []
        self.det_xy = np.zeros((0, 2))
        self.det_ok = np.zeros(0, dtype=bool)
        self.tracks = []

    def step(self, image: np.ndarray, timestamp: float, capture_monotonic: float) -> None:
        if self.localizer.due(timestamp):
            self.localizer.process(image, timestamp)
        pose = self.localizer.pose
        pose_valid = self.localizer.pose_valid(timestamp)

        self.detections = self.detector.detect(image) if self.detector is not None else []

        measurements: list[Measurement] = []
        if pose_valid and self.detections:
            pix = np.array([d.bottom_center for d in self.detections])
            xy, ok = project_pixels_to_floor(
                pix, self.intr.camera_matrix, self.intr.dist_coeffs, pose,
                floor_z=self.cfg.projection.floor_z_m, ground_offset_m=self.cfg.projection.ground_offset_m,
            )
            ok &= self.field.in_bounds(xy, self.cfg.field.bounds_margin_m)
            measurements = [
                Measurement(float(xy[i, 0]), float(xy[i, 1]), d.class_id, d.confidence)
                for i, d in enumerate(self.detections) if ok[i]
            ]
            self.det_xy, self.det_ok = xy, ok
        else:
            self.det_xy = np.full((len(self.detections), 2), np.nan)
            self.det_ok = np.zeros(len(self.detections), dtype=bool)

        self.tracks = self.tracker.update(measurements)
        self.publisher.publish(
            [t.x for t in self.tracks],
            [t.y for t in self.tracks],
            capture_monotonic,
            pose.as_list() if pose is not None else [0.0] * 6,
            pose_valid,
        )


def build_pipeline(cfg: Config, args: argparse.Namespace):
    calib_path = cfg.resolve_path(args.calib or cfg.camera.calibration_file)
    intr = load_intrinsics(calib_path)
    log.info("loaded calibration %s (%dx%d, rms %s px)", calib_path, intr.width, intr.height, intr.rms_px)

    layout = cfg.field.layout
    layout_path = cfg.resolve_path(layout)
    field = FieldLayout.load(
        str(layout_path) if layout_path.exists() else layout, cfg.field.tag_size_m, cfg.field.origin, cfg.field.ignore_tag_ids
    )
    log.info("field layout %s: %.2f x %.2f m, %d tags", layout, field.length, field.width, len(field.tags))

    from .detector import YoloDetector

    model_path = cfg.resolve_path(args.model or cfg.detector.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path} (set detector.model_path)")
    detector = YoloDetector(cfg.detector, str(model_path))

    if args.no_nt or not cfg.networktables.enabled:
        publisher = NullPublisher()
        log.info("NetworkTables publishing disabled")
    else:
        nt = cfg.networktables
        publisher = NTPublisher(cfg.team_number, nt.table, nt.server, nt.client_name)
    return intr, field, detector, publisher


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    show = cfg.debug.show_window if args.window is None else args.window

    intr, field, detector, publisher = build_pipeline(cfg, args)
    if args.replay:
        source = VideoSource(args.replay, speed=args.speed, loop=args.loop)
    else:
        source = CameraSource(cfg.camera)

    mapper: FieldMapper | None = None
    view = None
    fps_times: deque[float] = deque(maxlen=30)
    paused = False
    frame = None
    try:
        while True:
            if not paused or frame is None:
                frame = source.read()
                if frame is None:
                    if args.replay:
                        log.info("end of replay")
                        break
                    log.warning("no frame from camera")
                    continue

                if mapper is None:
                    h, w = frame.image.shape[:2]
                    intr = intr.scaled_to(w, h)
                    mapper = FieldMapper(cfg, intr, field, detector, publisher)
                    if show:
                        from .debug_view import DebugView

                        view = DebugView(field, intr.camera_matrix, intr.dist_coeffs, (w, h),
                                         cfg.debug.field_px_per_m, cfg.debug.camera_window_scale)

                mapper.step(frame.image, frame.timestamp, frame.capture_monotonic)
                fps_times.append(time.monotonic())

            if view is not None and mapper is not None:
                key = _show(view, mapper, frame, fps_times, publisher, paused, wait_ms=30 if paused else 1)
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    log.info("forcing re-localisation")
                    mapper.localizer.force()
                if key == ord(" ") and args.replay:
                    paused = not paused
                    if not paused:
                        source.resync()
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        publisher.close()
        if view is not None:
            cv2.destroyAllWindows()
    return 0


def _show(view, mapper: FieldMapper, frame, fps_times, publisher, paused, wait_ms: int = 1) -> int:
    loc = mapper.localizer
    now = frame.timestamp
    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 and fps_times[-1] > fps_times[0] else 0.0
    valid = loc.pose_valid(now)
    last = loc.last_result
    status = [
        f"{fps:5.1f} fps   pieces {len(mapper.tracks)}   detections {len(mapper.detections)}   "
        f"NT {'connected' if publisher.connected else 'not connected'}{'   [PAUSED]' if paused else ''}",
        f"pose {'VALID' if valid else 'INVALID'}"
        + (f"   last solve {now - last.timestamp:4.1f}s ago: {last.reason}" if last else "   (not localised yet)"),
    ]
    used = last.solution.tag_ids if last and last.solution and last.accepted else []
    cam = view.render_camera(frame.image, now, loc.pose, valid, last, mapper.detections, mapper.det_xy, mapper.det_ok, status)
    fld = view.render_field(loc.pose, valid, mapper.det_xy[mapper.det_ok], mapper.tracks, used)
    cv2.imshow("FieldMapper - camera", cam)
    cv2.imshow("FieldMapper - field", fld)
    return cv2.waitKey(wait_ms) & 0xFF
