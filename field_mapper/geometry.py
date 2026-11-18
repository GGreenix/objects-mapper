"""Camera pose maths and pixel-to-floor projection.

Frames used throughout:

* **Field frame** (WPILib convention): origin at the blue-alliance right-hand
  corner of the field, +X down the field towards the red wall, +Y to the left
  when standing at the blue driver station, +Z up. Metres.
* **OpenCV camera frame**: origin at the optical centre, +X right in the image,
  +Y down in the image, +Z out of the lens. This is what solvePnP works in.
* **Camera body frame** (WPILib convention, only used for the published
  roll/pitch/yaw): +X out of the lens, +Y left, +Z up.

A :class:`CameraPose` stores ``rotation`` (3x3, maps OpenCV-camera vectors into
the field frame) and ``position`` (optical centre in field metres).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Columns are the WPILib body axes (forward, left, up) expressed in OpenCV
# camera coordinates. ``R_field_body = R_field_cv @ CV_FROM_BODY``.
CV_FROM_BODY = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extrinsic X-Y-Z (roll, pitch, yaw) angles in radians, i.e. the angles
    WPILib's ``Rotation3d(roll, pitch, yaw)`` takes. ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    r = np.asarray(rotation, dtype=float)
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    pitch = float(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    return roll, pitch, yaw


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Inverse of :func:`rotation_to_rpy`."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quaternion_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Returns (w, x, y, z) with w >= 0."""
    r = np.asarray(rotation, dtype=float)
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def slerp_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation from ``r0`` (t=0) to ``r1`` (t=1)."""
    q0 = rotation_to_quaternion(r0)
    q1 = rotation_to_quaternion(r1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        q = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    q /= np.linalg.norm(q)
    return quaternion_to_rotation(*q)


def rotation_angle_between(r0: np.ndarray, r1: np.ndarray) -> float:
    """Angle (radians) of the relative rotation between two rotation matrices."""
    rel = np.asarray(r0).T @ np.asarray(r1)
    c = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Camera pose
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraPose:
    rotation: np.ndarray  # 3x3, field_from_opencv_camera
    position: np.ndarray  # (3,), optical centre in field metres

    @classmethod
    def from_rvec_tvec(cls, rvec: np.ndarray, tvec: np.ndarray) -> "CameraPose":
        """Build from solvePnP output (which maps field points into the camera frame)."""
        r_cam_field, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
        t = np.asarray(tvec, dtype=float).reshape(3)
        r_field_cam = r_cam_field.T
        return cls(rotation=r_field_cam, position=-r_field_cam @ t)

    def to_rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`from_rvec_tvec`; suitable for cv2.projectPoints."""
        r_cam_field = self.rotation.T
        rvec, _ = cv2.Rodrigues(r_cam_field)
        tvec = -r_cam_field @ self.position
        return rvec.reshape(3, 1), tvec.reshape(3, 1)

    @classmethod
    def look_at(
        cls,
        position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray | None = None,
        roll: float = 0.0,
    ) -> "CameraPose":
        """Camera at ``position`` with its optical axis pointing at ``target`` (``up`` defaults to +Z)."""
        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)
        position = np.asarray(position, dtype=float)
        z = np.asarray(target, dtype=float) - position
        z /= np.linalg.norm(z)
        x = np.cross(z, up)  # image right
        if np.linalg.norm(x) < 1e-9:  # looking straight up/down: pick any right vector
            x = np.cross(z, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(x) < 1e-9:
                x = np.cross(z, np.array([1.0, 0.0, 0.0]))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # image down
        rot = np.column_stack([x, y, z])
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            rot = rot @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return cls(rotation=rot, position=position)

    def body_rotation(self) -> np.ndarray:
        """Rotation of the WPILib camera body frame (x fwd, y left, z up) in the field."""
        return self.rotation @ CV_FROM_BODY

    def rpy(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in radians, WPILib convention (positive pitch = nose down)."""
        return rotation_to_rpy(self.body_rotation())

    def as_list(self) -> list[float]:
        """[x, y, z, roll, pitch, yaw] — metres and radians."""
        return [float(v) for v in self.position] + list(self.rpy())

    def distance_to(self, other: "CameraPose") -> tuple[float, float]:
        """(translation metres, rotation radians) between two poses."""
        return (
            float(np.linalg.norm(self.position - other.position)),
            rotation_angle_between(self.rotation, other.rotation),
        )

    def blend(self, other: "CameraPose", alpha: float) -> "CameraPose":
        """Move ``alpha`` (0..1) of the way from this pose towards ``other``."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return CameraPose(
            rotation=slerp_rotation(self.rotation, other.rotation, alpha),
            position=(1.0 - alpha) * self.position + alpha * other.position,
        )


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def undistort_to_normalized(pixels: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Distorted pixel coordinates (N, 2) -> ideal normalised image coords (N, 2)."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    k = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    try:  # OpenCV >= 5 accepts termination criteria directly
        out = cv2.undistortPoints(pts, k, d, None, None, None, _UNDISTORT_CRITERIA)
    except (cv2.error, TypeError):
        if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
            out = cv2.undistortPointsIter(pts, k, d, None, None, _UNDISTORT_CRITERIA)
        else:
            out = cv2.undistortPoints(pts, k, d)
    return out.reshape(-1, 2)


def pixel_rays(
    pixels: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, pose: CameraPose
) -> np.ndarray:
    """Unit ray directions in the field frame (N, 3) for distorted pixels (N, 2)."""
    norm = undistort_to_normalized(pixels, camera_matrix, dist_coeffs)
    rays_cam = np.column_stack([norm, np.ones(len(norm))])
    rays_field = rays_cam @ pose.rotation.T
    return rays_field / np.linalg.norm(rays_field, axis=1, keepdims=True)


def intersect_plane_z(
    origin: np.ndarray, directions: np.ndarray, plane_z: float = 0.0, min_down: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays ``origin + s * d`` (s > 0) with the horizontal plane ``z = plane_z``.

    Returns ``(points (N, 3), valid (N,))``. Rays that point upwards, are
    (near) parallel to the plane, or would hit it behind the camera are invalid
    and their points are NaN.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    height = origin[2] - plane_z
    dz = d[:, 2]
    # Camera above the plane needs rays going down (and vice versa).
    valid = -dz * np.sign(height) > min_down if height != 0 else np.zeros(len(d), dtype=bool)
    s = np.full(len(d), np.nan)
    s[valid] = (plane_z - origin[2]) / dz[valid]
    valid &= s > 0
    s[~valid] = np.nan
    return origin + s[:, None] * d, valid


def project_pixels_to_floor(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pose: CameraPose,
    floor_z: float = 0.0,
    ground_offset_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map distorted pixels (N, 2) to field (x, y) on the floor plane.

    ``ground_offset_m`` pushes each point further away from the camera along
    the ground, which turns "near edge of the object" (what the bottom of a
    bounding box sees) into roughly "centre of the object".

    Returns ``(xy (N, 2), valid (N,))``; invalid rows are NaN.
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(pixels) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    rays = pixel_rays(pixels, camera_matrix, dist_coeffs, pose)
    pts, valid = intersect_plane_z(pose.position, rays, floor_z)
    xy = pts[:, :2].copy()
    if ground_offset_m and np.any(valid):
        horiz = xy[valid] - pose.position[:2]
        dist = np.linalg.norm(horiz, axis=1, keepdims=True)
        dist[dist < 1e-9] = 1.0  # point directly below camera: no defined direction
        xy[valid] += ground_offset_m * horiz / dist
    return xy, valid


def in_field_bounds(xy: np.ndarray, length: float, width: float, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside ``[-m, L+m] x [-m, W+m]``. NaN rows are False."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return (
            (xy[:, 0] >= -margin)
            & (xy[:, 0] <= length + margin)
            & (xy[:, 1] >= -margin)
            & (xy[:, 1] <= width + margin)
        )


def project_field_points(
    points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, pose: CameraPose
) -> tuple[np.ndarray, np.ndarray]:
    """Field points (N, 3) -> pixels (N, 2) plus mask of points in front of the camera."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    in_cam = (pts - pose.position) @ pose.rotation  # rows: R^T (p - c)
    in_front = in_cam[:, 2] > 1e-3
    rvec, tvec = pose.to_rvec_tvec()
    px, _ = cv2.projectPoints(pts, rvec, tvec, np.asarray(camera_matrix, float), np.asarray(dist_coeffs, float))
    return px.reshape(-1, 2), in_front
