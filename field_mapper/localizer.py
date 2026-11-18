"""Camera self-localisation from field AprilTags.

* :class:`TagDetector` wraps robotpy-apriltag.
* :func:`solve_camera_pose` runs solvePnP over every visible tag's corners at once.
* :class:`PoseFilter` decides whether a new solve is trustworthy and blends it
  into the held pose (no snapping, outlier rejection, recovery if the camera
  really was moved).
* :class:`Localizer` ties them together on a timer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AprilTagConfig, LocalizationConfig
from .field import FieldLayout
from .geometry import CameraPose

log = logging.getLogger(__name__)


@dataclass
class TagObservation:
    tag_id: int
    corners: np.ndarray  # (4, 2) pixels, detector order (BL, BR, TR, TL)
    decision_margin: float = 0.0
    hamming: int = 0


@dataclass
class PoseSolution:
    pose: CameraPose
    reprojection_error_px: float  # RMS over all corners used
    tag_ids: list[int]
    ambiguity: float = 0.0  # single-tag only: best/alternative error ratio


@dataclass
class LocalizationResult:
    timestamp: float
    observations: list[TagObservation] = field(default_factory=list)
    solution: PoseSolution | None = None
    accepted: bool = False
    reason: str = ""


# --------------------------------------------------------------------------- #
class TagDetector:
    def __init__(self, cfg: AprilTagConfig, family: str = "tag36h11") -> None:
        import robotpy_apriltag as ra

        self.cfg = cfg
        self._det = ra.AprilTagDetector()
        if not self._det.addFamily(family, cfg.bits_corrected):
            raise ValueError(f"unknown AprilTag family '{family}'")
        c = self._det.getConfig()
        c.quadDecimate = float(cfg.quad_decimate)
        c.quadSigma = float(cfg.quad_sigma)
        c.numThreads = int(cfg.num_threads)
        c.refineEdges = bool(cfg.refine_edges)
        c.decodeSharpening = float(cfg.decode_sharpening)
        self._det.setConfig(c)
        q = self._det.getQuadThresholdParameters()
        q.minClusterPixels = int(cfg.min_cluster_pixels)
        self._det.setQuadThresholdParameters(q)

    def detect(self, gray: np.ndarray) -> list[TagObservation]:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray)
        out: list[TagObservation] = []
        for d in self._det.detect(gray):
            if d.getHamming() > self.cfg.max_hamming:
                continue
            if d.getDecisionMargin() < self.cfg.min_decision_margin:
                continue
            corners = np.array([[d.getCorner(i).x, d.getCorner(i).y] for i in range(4)], dtype=np.float64)
            out.append(TagObservation(d.getId(), corners, d.getDecisionMargin(), d.getHamming()))
        return out


# --------------------------------------------------------------------------- #
def _rms_reprojection(obj: np.ndarray, img: np.ndarray, rvec, tvec, k, d) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def solve_camera_pose(
    observations: list[TagObservation],
    field: FieldLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prior: CameraPose | None = None,
    min_tags: int = 1,
) -> PoseSolution | None:
    """Solve the camera's field pose from every known tag in view.

    Multiple tags: SQPnP (globally optimal) refined with Levenberg-Marquardt.
    Single tag: IPPE gives the two classic planar solutions; we drop any that
    put the camera under the floor, then prefer the one closest to the held
    pose (if any) or the lowest reprojection error.
    """
    used = [o for o in observations if o.tag_id in field]
    if len(used) < max(1, min_tags):
        return None
    obj = np.concatenate([field.corners(o.tag_id) for o in used]).astype(np.float64)
    img = np.concatenate([o.corners for o in used]).astype(np.float64)
    k = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(dist_coeffs, dtype=np.float64)
    ids = [o.tag_id for o in used]

    if len(used) == 1:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, k, d, flags=cv2.SOLVEPNP_IPPE)
        cands = []
        for i in range(n):
            rv, tv = cv2.solvePnPRefineLM(obj, img, k, d, rvecs[i].copy(), tvecs[i].copy())
            pose = CameraPose.from_rvec_tvec(rv, tv)
            if pose.position[2] < 0:
                continue  # camera below the floor: the mirror solution
            cands.append((_rms_reprojection(obj, img, rv, tv, k, d), pose))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        ambiguity = cands[0][0] / max(cands[1][0], 1e-9) if len(cands) > 1 else 0.0
        best_err, best_pose = cands[0]
        if prior is not None and len(cands) > 1:
            # Pick the solution nearest the pose we already trust.
            best_err, best_pose = min(cands, key=lambda c: c[1].distance_to(prior)[0])
        return PoseSolution(best_pose, best_err, ids, ambiguity)

    ok, rvec, tvec = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, d, rvec, tvec)
    err = _rms_reprojection(obj, img, rvec, tvec, k, d)
    return PoseSolution(CameraPose.from_rvec_tvec(rvec, tvec), err, ids, 0.0)


# --------------------------------------------------------------------------- #
class PoseFilter:
    """Smoothing + outlier rejection for successive pose solves."""

    def __init__(self, cfg: LocalizationConfig) -> None:
        self.cfg = cfg
        self.pose: CameraPose | None = None
        self.last_accept_time: float | None = None
        self.last_error_px: float | None = None
        self._pending: list[CameraPose] = []  # consecutive consistent jump candidates

    @property
    def confirming_jump(self) -> bool:
        return bool(self._pending)

    def is_valid(self, now: float) -> bool:
        if self.pose is None:
            return False
        if self.cfg.max_pose_age_s > 0 and self.last_accept_time is not None:
            return (now - self.last_accept_time) <= self.cfg.max_pose_age_s
        return True

    def reset(self) -> None:
        self.pose = None
        self.last_accept_time = None
        self._pending.clear()

    def update(self, sol: PoseSolution, now: float) -> tuple[bool, str]:
        cfg = self.cfg
        if sol.reprojection_error_px > cfg.max_reprojection_error_px:
            return False, f"reprojection error {sol.reprojection_error_px:.2f}px > {cfg.max_reprojection_error_px}px"
        if len(sol.tag_ids) == 1 and self.pose is None and sol.ambiguity > cfg.max_single_tag_ambiguity:
            return False, f"single-tag solve is ambiguous ({sol.ambiguity:.2f})"

        if self.pose is None:
            self._accept(sol.pose, sol, now)
            return True, "initial pose"

        dt, dr = self.pose.distance_to(sol.pose)
        if dt <= cfg.max_jump_m and np.degrees(dr) <= cfg.max_jump_deg:
            self._pending.clear()
            self._accept(self.pose.blend(sol.pose, cfg.smoothing_alpha), sol, now)
            return True, f"blended (moved {dt * 100:.1f}cm, {np.degrees(dr):.2f}deg)"

        # Looks like an outlier - unless several in a row agree with each other,
        # in which case the camera really moved.
        if cfg.jump_confirmations <= 0:
            return False, f"jump rejected ({dt:.2f}m, {np.degrees(dr):.1f}deg)"
        if self._pending:
            pdt, pdr = self._pending[-1].distance_to(sol.pose)
            if pdt > cfg.max_jump_m or np.degrees(pdr) > cfg.max_jump_deg:
                self._pending.clear()
        self._pending.append(sol.pose)
        if len(self._pending) >= cfg.jump_confirmations:
            new_pose = self._pending[0]
            for i, p in enumerate(self._pending[1:], start=2):
                new_pose = new_pose.blend(p, 1.0 / i)  # running average
            self._pending.clear()
            self._accept(new_pose, sol, now)
            return True, f"camera moved: re-anchored after {cfg.jump_confirmations} consistent solves"
        return False, (
            f"jump rejected ({dt:.2f}m, {np.degrees(dr):.1f}deg); "
            f"{len(self._pending)}/{cfg.jump_confirmations} towards re-anchor"
        )

    def _accept(self, pose: CameraPose, sol: PoseSolution, now: float) -> None:
        self.pose = pose
        self.last_accept_time = now
        self.last_error_px = sol.reprojection_error_px


# --------------------------------------------------------------------------- #
class Localizer:
    def __init__(
        self,
        loc_cfg: LocalizationConfig,
        tag_cfg: AprilTagConfig,
        field: FieldLayout,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_family: str = "tag36h11",
    ) -> None:
        self.cfg = loc_cfg
        self.field = field
        self.k = camera_matrix
        self.d = dist_coeffs
        self.detector = TagDetector(tag_cfg, tag_family)
        self.filter = PoseFilter(loc_cfg)
        self._next_time: float | None = None
        self.last_result: LocalizationResult | None = None

    @property
    def pose(self) -> CameraPose | None:
        return self.filter.pose

    def pose_valid(self, now: float) -> bool:
        return self.filter.is_valid(now)

    def due(self, now: float) -> bool:
        return self._next_time is None or now >= self._next_time

    def force(self) -> None:
        self._next_time = None

    def process(self, frame: np.ndarray, now: float) -> LocalizationResult:
        obs = self.detector.detect(frame)
        result = LocalizationResult(timestamp=now, observations=obs)
        sol = solve_camera_pose(obs, self.field, self.k, self.d, self.filter.pose, self.cfg.min_tags)
        result.solution = sol
        if sol is None:
            known = sum(1 for o in obs if o.tag_id in self.field)
            result.reason = f"not enough tags ({known} known, need {self.cfg.min_tags})"
        else:
            result.accepted, result.reason = self.filter.update(sol, now)

        held = self.filter.pose is not None and not self.filter.confirming_jump
        self._next_time = now + (self.cfg.interval_s if held else self.cfg.retry_interval_s)

        level = logging.INFO if result.accepted or self.filter.pose is None else logging.WARNING
        if sol is not None:
            log.log(
                level, "localise: tags=%s err=%.2fpx -> %s (%s)", sol.tag_ids, sol.reprojection_error_px,
                "accepted" if result.accepted else "rejected", result.reason,
            )
        else:
            log.debug("localise: %s", result.reason)
        self.last_result = result
        return result
