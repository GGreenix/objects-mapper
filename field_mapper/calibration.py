"""Load / save camera intrinsics (calib.yaml)."""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (N,)
    width: int
    height: int
    rms_px: float | None = None

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale for a different capture resolution with the same aspect ratio.

        Only valid if the camera scales (rather than crops) the sensor for the
        new mode. Recalibrating at the running resolution is always better.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 1e-3:
            raise ValueError(
                f"calibration is {self.width}x{self.height} but frames are {width}x{height} "
                "(different aspect ratio) - recalibrate at the capture resolution"
            )
        log.warning(
            "calibration is %dx%d, frames are %dx%d: scaling intrinsics. "
            "Recalibrate at the capture resolution for best accuracy.",
            self.width, self.height, width, height,
        )
        k = self.camera_matrix.copy()
        k[0, 0] *= sx
        k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
        k[1, 1] *= sy
        k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
        return CameraIntrinsics(k, self.dist_coeffs.copy(), width, height, self.rms_px)


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration file not found: {path} (run calibrate.py first)")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    k = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    return CameraIntrinsics(
        camera_matrix=k,
        dist_coeffs=d,
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        rms_px=data.get("rms_reprojection_error_px"),
    )


def save_intrinsics(path: str | Path, intr: CameraIntrinsics, extra: dict | None = None) -> None:
    data = {
        "image_width": int(intr.width),
        "image_height": int(intr.height),
        "camera_matrix": [[float(v) for v in row] for row in intr.camera_matrix],
        "dist_coeffs": [float(v) for v in intr.dist_coeffs.reshape(-1)],
        "rms_reprojection_error_px": None if intr.rms_px is None else float(intr.rms_px),
        "calibrated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        data.update(extra)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
