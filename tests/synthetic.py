"""Synthetic camera images of the field's AprilTags, for tests."""

from __future__ import annotations

import cv2
import numpy as np

from field_mapper.field import FieldLayout
from field_mapper.geometry import CameraPose

CELL_PX = 12  # marker image resolution per tag cell


def make_intrinsics(width: int = 1280, height: int = 720, hfov_deg: float = 70.0) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)


def tag_image(tag_id: int) -> np.ndarray:
    """36h11 tag (8x8 cells incl. black border) with a 1-cell white quiet zone -> 10x10 cells."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, tag_id, 8 * CELL_PX, borderBits=1)
    # OpenCV draws its AprilTag dictionaries rotated 180 deg from the official tag
    # images (checked against apriltag_to_image). Undo that so corner 0 is the
    # real bottom-left, as on a printed field tag.
    marker = np.ascontiguousarray(np.rot90(marker, 2))
    return cv2.copyMakeBorder(marker, CELL_PX, CELL_PX, CELL_PX, CELL_PX, cv2.BORDER_CONSTANT, value=255)


def render_tags(
    field: FieldLayout,
    pose: CameraPose,
    camera_matrix: np.ndarray,
    size: tuple[int, int] = (1280, 720),
    tag_ids: list[int] | None = None,
    background: int = 90,
    noise: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Render the tags visible from ``pose`` (pinhole, no distortion). Returns (BGR image, rendered ids)."""
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w), background, np.float32)
    rendered = []
    rvec, tvec = pose.to_rvec_tvec()
    ids = tag_ids if tag_ids is not None else list(field.tags)
    # far-to-near so nearer tags paint over farther ones
    ids = sorted(ids, key=lambda t: -np.linalg.norm(field.tags[t].position - pose.position))
    for tid in ids:
        tag = field.tags[tid]
        to_cam = pose.position - tag.position
        if np.dot(tag.rotation[:, 0], to_cam) / np.linalg.norm(to_cam) < 0.25:
            continue  # facing away or too oblique
        s = field.tag_size * 10 / 8 / 2  # half-size including the white quiet zone
        local = np.array([[0, -s, s], [0, s, s], [0, s, -s], [0, -s, -s]])  # TL, TR, BR, BL as viewed
        world = local @ tag.rotation.T + tag.position
        cam = (world - pose.position) @ pose.rotation
        if np.any(cam[:, 2] <= 0.1):
            continue
        px, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, None)
        px = px.reshape(-1, 2).astype(np.float32)
        if np.any(px < -w) or np.any(px[:, 0] > 2 * w) or np.any(px[:, 1] > 2 * h):
            continue
        timg = tag_image(tid)
        n = timg.shape[0]
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32) - 0.5
        H = cv2.getPerspectiveTransform(src, px)
        warped = cv2.warpPerspective(timg.astype(np.float32), H, (w, h), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.ones_like(timg, np.float32), H, (w, h), flags=cv2.INTER_LINEAR)
        img = img * (1 - mask) + warped * mask
        rendered.append(tid)
    img += rng.normal(0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), rendered


def render_checkerboard_views(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    size: tuple[int, int],
    pattern: tuple[int, int] = (9, 6),
    square_m: float = 0.025,
    n_views: int = 16,
    seed: int = 1,
) -> list[np.ndarray]:
    """Grey images of a flat checkerboard in random poses, rendered *with* lens distortion
    (each output pixel is undistorted, cast onto the board plane and sampled)."""
    w, h = size
    cols, rows = pattern
    s, m = 40, 40  # board texture: px per square, margin
    board = np.full(((rows + 1) * s + 2 * m, (cols + 1) * s + 2 * m), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[m + r * s: m + (r + 1) * s, m + c * s: m + (c + 1) * s] = 0
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10)
    try:
        norm = cv2.undistortPoints(pix, camera_matrix, dist_coeffs, None, None, None, crit)
    except (cv2.error, TypeError):
        norm = cv2.undistortPointsIter(pix, camera_matrix, dist_coeffs, None, None, crit)
    rays = np.column_stack([norm.reshape(-1, 2), np.ones(w * h)])
    rng = np.random.default_rng(seed)
    centre_local = np.array([(cols - 1) * square_m / 2, (rows - 1) * square_m / 2, 0.0])
    views = []
    while len(views) < n_views:
        rot, _ = cv2.Rodrigues(rng.normal(0, 0.35, 3))
        z = rng.uniform(0.35, 0.7)
        centre = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.15, 0.15) * z, z])
        t = centre - rot @ centre_local
        normal = rot[:, 2]
        denom = rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (normal @ t) / denom
        local = (rays * scale[:, None] - t) @ rot  # board-plane coordinates, origin at first inner corner
        mapx = ((local[:, 0] + square_m) / square_m * s + m).reshape(h, w).astype(np.float32)
        mapy = ((local[:, 1] + square_m) / square_m * s + m).reshape(h, w).astype(np.float32)
        img = cv2.remap(board, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
        img[~(scale.reshape(h, w) > 0)] = 128
        views.append(np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8))
    return views
