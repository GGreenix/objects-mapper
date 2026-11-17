"""Frame sources: a live USB camera (threaded, always-latest-frame) and video replay."""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)

_BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
}


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp: float  # pipeline clock (s): monotonic for live, video time for replay
    capture_monotonic: float  # time.monotonic() when the frame was obtained (for latency)


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    backend = cfg.backend.lower()
    if backend == "auto":
        backend = "dshow" if sys.platform.startswith("win") else ("v4l2" if sys.platform.startswith("linux") else "any")
    if backend not in _BACKENDS:
        raise ValueError(f"camera.backend must be one of auto, {', '.join(_BACKENDS)}")
    cap = cv2.VideoCapture(int(cfg.index), _BACKENDS[backend])
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {cfg.index} with backend {backend}")
    if cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for name, value in (cfg.properties or {}).items():
        prop = getattr(cv2, f"CAP_PROP_{str(name).upper()}", None)
        if prop is None:
            log.warning("unknown camera property %s (expected a cv2.CAP_PROP_* suffix)", name)
            continue
        if not cap.set(prop, float(value)):
            log.warning("camera rejected %s=%s", name, value)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if (w, h) != (cfg.width, cfg.height):
        log.warning("camera gave %dx%d instead of requested %dx%d", w, h, cfg.width, cfg.height)
    log.info("camera %d open: %dx%d @ %.1f fps (%s)", cfg.index, w, h, fps, backend)
    return cap


class CameraSource:
    """Grabs frames on a background thread and keeps only the newest one, so a
    slow consumer never works on stale, buffered frames."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cap = open_camera(cfg)
        self._lock = threading.Condition()
        self._frame: Frame | None = None
        self._last_read_index = -1
        self._running = True
        self._failures = 0
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        index = 0
        while self._running:
            ok, img = self.cap.read()
            t = time.monotonic()
            if not ok or img is None:
                self._failures += 1
                if self._failures in (1, 30) or self._failures % 300 == 0:
                    log.warning("camera read failed (%d in a row)", self._failures)
                time.sleep(0.01)
                continue
            self._failures = 0
            with self._lock:
                self._frame = Frame(img, index, t, t)
                self._lock.notify_all()
            index += 1

    def read(self, timeout: float = 2.0) -> Frame | None:
        """Blocks until a frame newer than the last one returned is available."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._running and (self._frame is None or self._frame.index == self._last_read_index):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)
            if self._frame is None:
                return None
            self._last_read_index = self._frame.index
            return self._frame

    def close(self) -> None:
        self._running = False
        with self._lock:
            self._lock.notify_all()
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoSource:
    """Replays a video file. ``speed`` 1.0 = real time, 0 = as fast as possible.
    Timestamps follow the video clock, so time-based behaviour (re-localisation
    every N seconds) matches what would have happened live."""

    def __init__(self, path: str | Path, speed: float = 1.0, loop: bool = False) -> None:
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"replay file not found: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.speed = max(0.0, float(speed))
        self.loop = loop
        self.paused = False
        self._index = 0
        self._loop_offset = 0.0
        self._t0: float | None = None
        self._video_t0 = 0.0
        log.info("replaying %s (%.1f fps, speed %s)", self.path, self.fps, self.speed or "max")

    def read(self, timeout: float = 0.0) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._loop_offset = self._index / self.fps
            ok, img = self.cap.read()
            if not ok:
                return None
            self._t0 = None
        video_t = self._loop_offset + (self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / self.fps
        if self.speed > 0:
            now = time.monotonic()
            if self._t0 is None:
                self._t0, self._video_t0 = now, video_t
            target = self._t0 + (video_t - self._video_t0) / self.speed
            if target > now:
                time.sleep(target - now)
        frame = Frame(img, self._index, video_t, time.monotonic())
        self._index += 1
        return frame

    def resync(self) -> None:
        """Call after a pause so pacing doesn't try to catch up."""
        self._t0 = None

    def close(self) -> None:
        self.cap.release()
