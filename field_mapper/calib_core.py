"""Checkerboard calibration: corner detection, solving, and an interactive
session used by the web UI (calibrate.py uses the same functions)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraIntrinsics, save_intrinsics

log = logging.getLogger(__name__)

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Accurate full-resolution corner detection. Returns (N, 2) or None."""
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners.reshape(-1, 2).astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners.reshape(-1, 2).astype(np.float32)


def quick_corners(gray: np.ndarray, pattern: tuple[int, int], max_width: int = 640) -> np.ndarray | None:
    """Fast, approximate detection on a downscaled image (for live preview)."""
    scale = min(1.0, max_width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    return corners.reshape(-1, 2) / scale if ok else None


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def calibrate(
    views: list[np.ndarray], pattern: tuple[int, int], square: float, size: tuple[int, int], rational: bool = False
) -> tuple[CameraIntrinsics, list[float], int]:
    """cv2.calibrateCamera, then drop clearly bad views (> 3x median error) once and re-solve.

    Returns (intrinsics, per-view RMS errors, number of views dropped).
    """
    obj = board_points(pattern, square)
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0

    def solve(vs):
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
            [obj] * len(vs), [v.reshape(-1, 1, 2) for v in vs], size, None, None, flags=flags
        )
        errs = []
        for v, r, t in zip(vs, rvecs, tvecs, strict=True):
            proj, _ = cv2.projectPoints(obj, r, t, k, d)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v) ** 2, axis=1)))))
        return rms, k, d, errs

    rms, k, d, errs = solve(views)
    limit = max(3 * float(np.median(errs)), 0.5)
    keep = [v for v, e in zip(views, errs, strict=True) if e <= limit]
    dropped = len(views) - len(keep)
    if dropped and len(keep) >= 5:
        log.info("dropping %d view(s) with error > %.2fpx and re-solving", dropped, limit)
        rms, k, d, errs = solve(keep)
    else:
        dropped = 0
    return CameraIntrinsics(k, d.reshape(-1), size[0], size[1], float(rms)), errs, dropped


def summarize(intr: CameraIntrinsics, errs: list[float], dropped: int) -> dict:
    k = intr.camera_matrix
    rms = float(intr.rms_px or 0.0)
    quality = "excellent" if rms < 0.3 else "good" if rms < 0.6 else "usable" if rms < 1.0 else "poor"
    return {
        "rms_px": rms,
        "worst_view_px": float(max(errs)) if errs else 0.0,
        "views_used": len(errs),
        "views_dropped": dropped,
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "hfov_deg": float(np.degrees(2 * np.arctan(intr.width / (2 * k[0, 0])))),
        "vfov_deg": float(np.degrees(2 * np.arctan(intr.height / (2 * k[1, 1])))),
        "dist": [float(v) for v in intr.dist_coeffs],
        "width": intr.width,
        "height": intr.height,
        "quality": quality,
    }


def is_new_view(c: np.ndarray, views: list[np.ndarray], diag: float, min_diff: float = 0.05) -> bool:
    """True if corners ``c`` differ enough from every previous view (mean corner shift / image diagonal)."""
    if not views:
        return True
    return min(float(np.mean(np.linalg.norm(c - v, axis=1))) / diag for v in views) > min_diff


def printable_board(cols: int, rows: int, square_mm: float, dpi: int = 300) -> np.ndarray:
    """A checkerboard with ``cols`` x ``rows`` inner corners, sized for printing at ``dpi``."""
    sq = int(round(square_mm / 25.4 * dpi))
    margin = sq
    w, h = (cols + 1) * sq + 2 * margin, (rows + 1) * sq + 2 * margin + sq
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    label = f"{cols}x{rows} inner corners, {square_mm:g} mm squares. Print at 100% (no 'fit to page'), then measure a square."
    scale = max(0.5, sq / 120)
    cv2.putText(img, label, (margin, h - int(sq * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(scale * 2)), cv2.LINE_AA)
    return img


class CalibrationSession:
    """Interactive calibration fed with live frames by the pipeline thread.

    Web requests only set flags / read state; all image work happens in
    :meth:`on_frame` (pipeline thread) or the solver thread.
    """

    def __init__(self, cols: int, rows: int, square_m: float, min_views: int, output: Path) -> None:
        self.pattern = (int(cols), int(rows))
        self.square = float(square_m)
        self.min_views = int(min_views)
        self.output = Path(output)
        self.views: list[np.ndarray] = []
        self._quick_views: list[np.ndarray] = []
        self.size: tuple[int, int] | None = None
        self.auto = False
        self.quick: np.ndarray | None = None
        self.still = False
        self.message = "Hold the board in front of the camera."
        self.result: dict | None = None
        self.solving = False
        self.saved = False
        self._prev_quick: np.ndarray | None = None
        self._last_quick_time = 0.0
        self._last_capture = 0.0
        self._capture_requested = False
        self._lock = threading.Lock()

    # ---- commands (any thread)
    def request_capture(self) -> None:
        self._capture_requested = True

    def set_auto(self, on: bool) -> None:
        self.auto = bool(on)

    def undo(self) -> None:
        with self._lock:
            if self.views:
                self.views.pop()
                self._quick_views.pop()
                self.message = f"Removed last view ({len(self.views)} left)."

    def clear(self) -> None:
        with self._lock:
            self.views.clear()
            self._quick_views.clear()
            self.result = None
            self.saved = False
            self.message = "Cleared."

    def start_solve(self) -> bool:
        with self._lock:
            if self.solving or len(self.views) < self.min_views or self.size is None:
                return False
            views, size = list(self.views), self.size
            self.solving = True
            self.message = f"Calibrating with {len(views)} views..."
        threading.Thread(target=self._solve, args=(views, size), name="calibrate", daemon=True).start()
        return True

    def _solve(self, views, size) -> None:
        try:
            intr, errs, dropped = calibrate(views, self.pattern, self.square, size)
            summary = summarize(intr, errs, dropped)
            save_intrinsics(
                self.output, intr,
                extra={"num_views": len(views) - dropped,
                       "board": {"cols": self.pattern[0], "rows": self.pattern[1], "square_size_m": self.square}},
            )
            log.info("calibration saved to %s (RMS %.3f px)", self.output, summary["rms_px"])
            with self._lock:
                self.result, self.saved = summary, True
                self.message = f"Saved. RMS error {summary['rms_px']:.2f} px ({summary['quality']})."
        except Exception as e:  # cv2 errors on degenerate input
            log.exception("calibration failed")
            with self._lock:
                self.message = f"Calibration failed: {e}"
        finally:
            self.solving = False

    # ---- pipeline thread
    def on_frame(self, image: np.ndarray) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        size = (gray.shape[1], gray.shape[0])
        if self.size != size:
            with self._lock:
                if self.size is not None and self.views:
                    self.message = "Resolution changed - views cleared."
                self.views.clear()
                self._quick_views.clear()
                self.size = size
        now = time.monotonic()
        want = self._capture_requested
        if want or now - self._last_quick_time > 0.12:  # preview detection ~8 Hz
            self._last_quick_time = now
            quick = quick_corners(gray, self.pattern)
            diag = float(np.hypot(*size))
            self.still = (
                quick is not None and self._prev_quick is not None
                and float(np.mean(np.linalg.norm(quick - self._prev_quick, axis=1))) < 0.003 * diag
            )
            self._prev_quick = quick
            self.quick = quick
            auto_ok = (
                self.auto and self.still and now - self._last_capture > 1.0
                and quick is not None and is_new_view(quick, self._quick_views, diag)
            )
            if want or auto_ok:
                self._capture_requested = False
                self._capture(gray, quick)

    def _capture(self, gray: np.ndarray, quick: np.ndarray | None) -> None:
        if quick is None:
            self.message = "No board found - make sure the whole board is visible."
            return
        corners = find_corners(gray, self.pattern)
        if corners is None:
            self.message = "Board seen but corners not precise - hold it still and try again."
            return
        with self._lock:
            self.views.append(corners)
            self._quick_views.append(quick)
            self._last_capture = time.monotonic()
            n = len(self.views)
            self.message = f"Captured view {n}." + (" Ready to calibrate." if n >= self.min_views else "")

    # ---- rendering / state
    def render(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        with self._lock:
            views = list(self.views)
        overlay = img.copy()
        for v in views:
            hull = cv2.convexHull(v.astype(np.float32)).astype(np.int32)
            cv2.fillPoly(overlay, [hull], (60, 180, 60))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        for v in views:
            for p in v[:: max(1, len(v) // 12)]:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (60, 220, 60), -1)
        if self.quick is not None:
            cv2.drawChessboardCorners(img, self.pattern, self.quick.reshape(-1, 1, 2).astype(np.float32), True)
        return img

    def state(self) -> dict:
        with self._lock:
            return {
                "active": True,
                "cols": self.pattern[0],
                "rows": self.pattern[1],
                "square_mm": self.square * 1000,
                "views": len(self.views),
                "min_views": self.min_views,
                "auto": self.auto,
                "board_seen": self.quick is not None,
                "still": self.still,
                "message": self.message,
                "solving": self.solving,
                "result": self.result,
                "saved": self.saved,
            }
