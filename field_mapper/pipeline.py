"""One frame of the pipeline: localise (on schedule) -> detect -> project -> track -> publish."""

from __future__ import annotations

import time

import numpy as np

from .calibration import CameraIntrinsics
from .config import Config
from .field import FieldLayout
from .geometry import project_pixels_to_floor
from .localizer import Localizer
from .tracker import Measurement, NearestNeighbourTracker


class FieldMapper:
    def __init__(self, cfg: Config, intrinsics: CameraIntrinsics, field: FieldLayout, detector, publisher) -> None:
        self.cfg = cfg
        self.intr = intrinsics
        self.field = field
        self.detector = detector  # may be None (no model yet): localisation still runs
        self.publisher = publisher
        self.localizer = Localizer(
            cfg.localization, cfg.apriltag, field, intrinsics.camera_matrix, intrinsics.dist_coeffs, cfg.field.tag_family
        )
        t = cfg.tracker
        self.tracker = NearestNeighbourTracker(t.max_match_distance_m, t.max_missed_frames, t.min_hits, t.smoothing_alpha)
        self.detections = []
        self.det_xy = np.zeros((0, 2))
        self.det_ok = np.zeros(0, dtype=bool)
        self.tracks = []
        self.timing = {"localize_ms": 0.0, "detect_ms": 0.0, "step_ms": 0.0}

    def step(self, image: np.ndarray, timestamp: float, capture_monotonic: float) -> None:
        t0 = time.perf_counter()
        if self.localizer.due(timestamp):
            self.localizer.process(image, timestamp)
            self.timing["localize_ms"] = (time.perf_counter() - t0) * 1000
        pose = self.localizer.pose
        pose_valid = self.localizer.pose_valid(timestamp)

        t1 = time.perf_counter()
        self.detections = self.detector.detect(image) if self.detector is not None else []
        self.timing["detect_ms"] = (time.perf_counter() - t1) * 1000 if self.detector is not None else 0.0

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
        self.timing["step_ms"] = (time.perf_counter() - t0) * 1000
