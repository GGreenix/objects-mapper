"""Debug drawing: camera overlay and a top-down field view."""

from __future__ import annotations

import numpy as np
import cv2

from .detector import Detection
from .field import FieldLayout
from .geometry import CameraPose, intersect_plane_z, pixel_rays, project_field_points, undistort_to_normalized
from .localizer import LocalizationResult
from .tracker import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
ORANGE = (0, 150, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
CLASS_COLOURS = [(0, 200, 255), (255, 160, 0), (200, 80, 255), (80, 255, 160), (255, 80, 120), (160, 255, 0)]


def _text(img, s, org, scale=0.5, colour=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _ipt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


class DebugView:
    def __init__(
        self,
        field: FieldLayout,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: tuple[int, int],
        px_per_m: float = 60.0,
        camera_scale: float = 1.0,
        margin_m: float = 1.5,
    ) -> None:
        self.field = field
        self.k = camera_matrix
        self.d = dist_coeffs
        self.w, self.h = image_size
        self.s = float(px_per_m)
        self.cam_scale = float(camera_scale)
        self.m = float(margin_m)
        self.fw = int((field.length + 2 * self.m) * self.s)
        self.fh = int((field.width + 2 * self.m) * self.s)
        self._background = self._draw_field_background()
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

    # ------------------------------------------------------------------ field
    def to_px(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x + self.m) * self.s)), int(round((self.field.width + self.m - y) * self.s))

    def _draw_field_background(self) -> np.ndarray:
        img = np.full((self.fh, self.fw, 3), 40, np.uint8)
        L, W = self.field.length, self.field.width
        cv2.rectangle(img, self.to_px(0, W), self.to_px(L, 0), (45, 75, 45), -1)
        for x in range(1, int(L) + 1):
            cv2.line(img, self.to_px(x, 0), self.to_px(x, W), (60, 95, 60), 1)
        for y in range(1, int(W) + 1):
            cv2.line(img, self.to_px(0, y), self.to_px(L, y), (60, 95, 60), 1)
        cv2.line(img, self.to_px(L / 2, 0), self.to_px(L / 2, W), (200, 200, 200), 1)
        cv2.line(img, self.to_px(0, 0), self.to_px(0, W), (230, 120, 40), 4)  # blue wall
        cv2.line(img, self.to_px(L, 0), self.to_px(L, W), (50, 50, 220), 4)  # red wall
        cv2.rectangle(img, self.to_px(0, W), self.to_px(L, 0), (200, 200, 200), 1)
        for tid, tag in self.field.tags.items():
            c = tag.position
            normal = tag.rotation[:, 0]
            side = tag.rotation[:, 1]
            a = c[:2] - side[:2] * 0.15
            b = c[:2] + side[:2] * 0.15
            cv2.line(img, self.to_px(*a), self.to_px(*b), WHITE, 2)
            tip = c[:2] + normal[:2] * 0.25
            cv2.line(img, self.to_px(*c[:2]), self.to_px(*tip), GREY, 1)
            lp = c[:2] + normal[:2] * 0.45
            px, py = self.to_px(*lp)
            cv2.putText(img, str(tid), (px - 7, py + 4), FONT, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
        return img

    def _footprint(self, pose: CameraPose, max_range: float = 25.0) -> np.ndarray:
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
        for p, ok, r in zip(pts, valid, rays):
            if ok and np.linalg.norm(p[:2] - cam) <= max_range:
                out.append(p[:2])
            else:
                hz = r[:2]
                if np.linalg.norm(hz) < 1e-6:
                    continue
                out.append(cam + hz / np.linalg.norm(hz) * max_range)
        return np.array(out)

    def render_field(
        self,
        pose: CameraPose | None,
        pose_valid: bool,
        raw_xy: np.ndarray,
        tracks: list[Track],
        used_tags: list[int] | None = None,
    ) -> np.ndarray:
        img = self._background.copy()
        for tid in used_tags or []:
            if tid in self.field.tags:
                cv2.circle(img, self.to_px(*self.field.tags[tid].position[:2]), 6, GREEN, 2)

        if pose is not None:
            fp = self._footprint(pose)
            if len(fp) >= 3:
                poly = np.array([self.to_px(*p) for p in fp], np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly], (90, 90, 40))
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [poly], True, (130, 130, 60), 1)
            cx, cy = self.to_px(*pose.position[:2])
            cx = int(np.clip(cx, 6, self.fw - 7))
            cy = int(np.clip(cy, 6, self.fh - 7))
            fwd = pose.body_rotation()[:, 0]
            hd = fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-6)
            tip = (int(cx + hd[0] * 30), int(cy - hd[1] * 30))
            colour = GREEN if pose_valid else RED
            cv2.arrowedLine(img, (cx, cy), tip, colour, 2, tipLength=0.3)
            cv2.circle(img, (cx, cy), 6, colour, -1)

        for p in raw_xy:
            cv2.circle(img, self.to_px(*p), 3, GREY, -1)
        r = max(4, int(0.09 * self.s))
        for t in tracks:
            colour = CLASS_COLOURS[t.class_id % len(CLASS_COLOURS)]
            p = self.to_px(t.x, t.y)
            cv2.circle(img, p, r, colour, -1 if t.misses == 0 else 2)
            _text(img, str(t.track_id), (p[0] + r + 2, p[1] - r), 0.4, colour)

        _text(img, f"pieces: {len(tracks)}", (8, 18), 0.5)
        if pose is not None:
            x, y, z, roll, pitch, yaw = pose.as_list()
            _text(
                img,
                f"cam ({x:.2f}, {y:.2f}, {z:.2f}) m  r/p/y ({np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f}) deg",
                (8, self.fh - 8), 0.45, GREEN if pose_valid else RED,
            )
        else:
            _text(img, "no camera pose", (8, self.fh - 8), 0.5, RED)
        return img

    # ----------------------------------------------------------------- camera
    def _project(self, pose: CameraPose, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, front = project_field_points(points, self.k, self.d, pose)
        cam = (np.asarray(points, float) - pose.position) @ pose.rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = front & (np.abs(cam[:, 0] / cam[:, 2]) < self._fov_x) & (np.abs(cam[:, 1] / cam[:, 2]) < self._fov_y)
        return px, ok

    def render_camera(
        self,
        frame: np.ndarray,
        now: float,
        pose: CameraPose | None,
        pose_valid: bool,
        loc: LocalizationResult | None,
        detections: list[Detection],
        det_xy: np.ndarray,
        det_ok: np.ndarray,
        status: list[str],
    ) -> np.ndarray:
        img = frame.copy()

        if pose is not None:
            # Where the layout says the field and tags should be, given our pose.
            px, ok = self._project(pose, self._perimeter)
            self._polyline_segments(img, px, ok, (200, 200, 200))
            for tid, tag in self.field.tags.items():
                if np.dot(tag.rotation[:, 0], pose.position - tag.position) <= 0:
                    continue  # tag faces away from the camera
                c, ok = self._project(pose, self.field.corners(tid))
                if ok.all():
                    cv2.polylines(img, [c.astype(np.int32)], True, YELLOW, 1, cv2.LINE_AA)

        if loc is not None and now - loc.timestamp < 1.5:
            used = set(loc.solution.tag_ids) if loc.solution else set()
            for o in loc.observations:
                colour = GREEN if (o.tag_id in used and loc.accepted) else ORANGE
                cv2.polylines(img, [o.corners.astype(np.int32)], True, colour, 2, cv2.LINE_AA)
                cv2.circle(img, _ipt(o.corners[0]), 4, colour, -1)  # corner 0 = bottom-left
                c = o.corners.mean(axis=0)
                _text(img, str(o.tag_id), _ipt(c), 0.6, colour, 2)

        for i, det in enumerate(detections):
            if not pose_valid:
                colour, where = RED, "no pose"
            elif det_ok[i]:
                colour, where = GREEN, f"({det_xy[i, 0]:.2f}, {det_xy[i, 1]:.2f})"
            else:
                colour, where = ORANGE, "off field"
            cv2.rectangle(img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), colour, 2)
            cv2.circle(img, _ipt(det.bottom_center), 4, colour, -1)
            _text(img, f"{det.class_name or det.class_id} {det.confidence:.2f} {where}", (int(det.x1), max(14, int(det.y1) - 6)), 0.45, colour)

        bar_h = 22 * len(status) + 8
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        for i, line in enumerate(status):
            _text(img, line, (8, 20 + 22 * i), 0.55)

        if self.cam_scale != 1.0:
            img = cv2.resize(img, None, fx=self.cam_scale, fy=self.cam_scale, interpolation=cv2.INTER_AREA)
        return img

    def _polyline_segments(self, img, px, front, colour) -> None:
        seg: list[tuple[int, int]] = []
        for p, f in zip(px, front):
            if f:
                seg.append(_ipt(p))
            elif len(seg) > 1:
                cv2.polylines(img, [np.array(seg, np.int32)], False, colour, 1, cv2.LINE_AA)
                seg = []
            else:
                seg = []
        if len(seg) > 1:
            cv2.polylines(img, [np.array(seg, np.int32)], False, colour, 1, cv2.LINE_AA)
