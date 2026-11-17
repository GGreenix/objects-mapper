"""Game-piece detection with an Ultralytics YOLO model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the piece meets the floor (approximately): bottom-centre of the box."""
        return (0.5 * (self.x1 + self.x2), self.y2)


class YoloDetector:
    def __init__(self, cfg: DetectorConfig, model_path: str) -> None:
        import torch
        from ultralytics import YOLO

        self.cfg = cfg
        device = str(cfg.device)
        if device.startswith("cuda") or device.isdigit():
            if not torch.cuda.is_available():
                log.warning("CUDA requested (%s) but not available - falling back to CPU. "
                            "Install a CUDA build of torch for GPU inference.", device)
                device = "cpu"
            else:
                log.info("YOLO on GPU: %s", torch.cuda.get_device_name(0))
        self.device = device
        self.half = bool(cfg.half) and device != "cpu"
        self.model = YOLO(model_path)
        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids = self._resolve_classes(cfg.classes)
        self._predict_kwargs = dict(
            conf=cfg.confidence, iou=cfg.iou, imgsz=cfg.imgsz, device=device, classes=self.class_ids, verbose=False
        )
        if self.half:
            # Newer Ultralytics replaced `half` with `quantize` (and warns on every call otherwise).
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT

                new_api = "quantize" in DEFAULT_CFG_DICT
            except ImportError:
                new_api = False
            self._predict_kwargs.update({"quantize": 16} if new_api else {"half": True})
        # Warm-up so the first real frame isn't slow (CUDA context, cudnn autotune).
        self.detect(np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8))
        log.info("YOLO model %s loaded (%d classes, device=%s, half=%s)", model_path, len(self.names), device, self.half)

    def set_thresholds(self, confidence: float, iou: float | None = None) -> None:
        self.cfg.confidence = float(confidence)
        self._predict_kwargs["conf"] = float(confidence)
        if iou is not None:
            self.cfg.iou = float(iou)
            self._predict_kwargs["iou"] = float(iou)

    def _resolve_classes(self, classes: list) -> list[int] | None:
        if not classes:
            return None
        by_name = {v.lower(): k for k, v in self.names.items()}
        ids = []
        for c in classes:
            if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
                ids.append(int(c))
            elif str(c).lower() in by_name:
                ids.append(by_name[str(c).lower()])
            else:
                raise ValueError(f"detector.classes: '{c}' not in model classes {list(self.names.values())}")
        return ids

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, **self._predict_kwargs)
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
            if c < self.cfg.confidence:
                continue
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), int(k), self.names.get(int(k), str(k))))
        return out
