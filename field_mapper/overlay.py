"""Camera-image overlay (drawn server-side for the web stream) and the camera's
floor footprint (drawn client-side on the field view)."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .field import FieldLayout
from .geometry import CameraPose, intersect_plane_z, pixel_rays, project_field_points, undistort_to_normalized
from .localizer import LocalizationResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
ORANGE = (0, 150, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)


def text(img, s, org, scale=0.5, colour=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def banner(img: np.ndarray, message: str, colour=(0, 140, 255)) -> np.ndarray:
    """Draw a centred message bar across the image (for 'calibration needed' etc.)."""
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.6, w / 1400)
    (tw, th), _ = cv2.getTextSize(message, FONT, scale, 2)
    y0 = h // 2 - th
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0 - th), (w, y0 + 2 * th), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    text(out, message, ((w - tw) // 2, y0 + th // 2), scale, colour, 2)
    return out


def _ipt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


class Overlay:
    def __init__(self, field: FieldLayout, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple[int, int]) -> None:
        self.field = field
        self.k = camera_matrix
        self.d = dist_coeffs
        self.w, self.h = image_size
        # Field perimeter sampled densely so it bends correctly under lens distortion.
        L, W = field.length, field.width
        n = 60
        edges = [
            np.column_stack([np.linspace(0, L, n), np.zeros(n)]),
            np.column_stack([np.full(n, L), np.linspace(0, W, n)]),
            np.column_stack([np.linspace(L, 0, n), np.full(n, W)]),
            np.column_stack([np.zeros(n), np.linspace(W, 0, n)]),
        ]
        self._perimeter = np.column_stack([np.concatenate(edges), np.zeros(4 * n)])
        # Points outside the lens' field of view must not be drawn: the distortion
        # polynomial folds them back into the image at nonsense positions.
        corners = np.array([[0, 0], [self.w - 1, 0], [0, self.h - 1], [self.w - 1, self.h - 1]], float)
        norm = np.abs(undistort_to_normalized(corners, camera_matrix, dist_coeffs))
        self._fov_x = float(norm[:, 0].max()) * 1.15
        self._fov_y = float(norm[:, 1].max()) * 1.15

    def footprint(self, pose: CameraPose, max_range: float = 25.0) -> np.ndarray:
        """Polygon (N, 2) of the floor area the camera sees, clipped at ``max_range`` metres."""
        n = 24
        w, h = self.w - 1, self.h - 1
        border = np.concatenate(
            [
                np.column_stack([np.linspace(0, w, n), np.full(n, h)]),  # bottom
                np.column_stack([np.full(n, w), np.linspace(h, 0, n)]),  # right
                np.column_stack([np.linspace(w, 0, n), np.zeros(n)]),  # top
                np.column_stack([np.zeros(n), np.linspace(0, h, n)]),  # left
            ]
        )
        rays = pixel_rays(border, self.k, self.d, pose)
        pts, valid = intersect_plane_z(pose.position, rays, 0.0)
        out = []
        cam = pose.position[:2]
        for p, ok, r in zip(pts, valid, rays, strict=True):
            if ok and np.linalg.norm(p[:2] - cam) <= max_range:
                out.append(p[:2])
            else:
                hz = r[:2]
                if np.linalg.norm(hz) < 1e-6:
                    continue
                out.append(cam + hz / np.linalg.norm(hz) * max_range)
        return np.array(out)

    def _project(self, pose: CameraPose, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, front = project_field_points(points, self.k, self.d, pose)
        cam = (np.asarray(points, float) - pose.position) @ pose.rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = front & (np.abs(cam[:, 0] / cam[:, 2]) < self._fov_x) & (np.abs(cam[:, 1] / cam[:, 2]) < self._fov_y)
        return px, ok

    def render(
        self,
        frame: np.ndarray,
        now: float,
        pose: CameraPose | None,
        pose_valid: bool,
        loc: LocalizationResult | None,
        detections: list[Detection],
        det_xy: np.ndarray,
        det_ok: np.ndarray,
        show_field: bool = True,
        show_detections: bool = True,
    ) -> np.ndarray:
        img = frame.copy()

        if show_field and pose is not None:
            # Where the layout says the field edge and tags should be, given our pose.
            px, ok = self._project(pose, self._perimeter)
            self._polyline_segments(img, px, ok, (200, 200, 200))
            for tid, tag in self.field.tags.items():
                if np.dot(tag.rotation[:, 0], pose.position - tag.position) <= 0:
                    continue  # tag faces away from the camera
                c, ok = self._project(pose, self.field.corners(tid))
                if ok.all():
                    cv2.polylines(img, [c.astype(np.int32)], True, YELLOW, 1, cv2.LINE_AA)

        if show_field and loc is not None and now - loc.timestamp < 1.5:
            used = set(loc.solution.tag_ids) if loc.solution else set()
            for o in loc.observations:
                colour = GREEN if (o.tag_id in used and loc.accepted) else ORANGE
                cv2.polylines(img, [o.corners.astype(np.int32)], True, colour, 2, cv2.LINE_AA)
                cv2.circle(img, _ipt(o.corners[0]), 4, colour, -1)  # corner 0 = bottom-left
                text(img, str(o.tag_id), _ipt(o.corners.mean(axis=0)), 0.6, colour, 2)

        if show_detections:
            for i, det in enumerate(detections):
                if not pose_valid:
                    colour, where = RED, "no pose"
                elif i < len(det_ok) and det_ok[i]:
                    colour, where = GREEN, f"({det_xy[i, 0]:.2f}, {det_xy[i, 1]:.2f})"
                else:
                    colour, where = ORANGE, "off field"
                cv2.rectangle(img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), colour, 2)
                cv2.circle(img, _ipt(det.bottom_center), 4, colour, -1)
                label = f"{det.class_name or det.class_id} {det.confidence:.2f} {where}"
                text(img, label, (int(det.x1), max(14, int(det.y1) - 6)), 0.45, colour)
        return img

    def _polyline_segments(self, img, px, visible, colour) -> None:
        seg: list[tuple[int, int]] = []
        for p, ok in zip(px, visible, strict=True):
            if ok:
                seg.append(_ipt(p))
                continue
            if len(seg) > 1:
                cv2.polylines(img, [np.array(seg, np.int32)], False, colour, 1, cv2.LINE_AA)
            seg = []
        if len(seg) > 1:
            cv2.polylines(img, [np.array(seg, np.int32)], False, colour, 1, cv2.LINE_AA)
