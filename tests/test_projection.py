"""Floor projection maths: pixel -> ray -> z=0 intersection, and the pose conventions behind it."""

import cv2
import numpy as np
import pytest

from field_mapper.geometry import (
    CameraPose,
    in_field_bounds,
    intersect_plane_z,
    pixel_rays,
    project_field_points,
    project_pixels_to_floor,
    rotation_to_rpy,
    rpy_to_rotation,
    undistort_to_normalized,
)

W, H = 1280, 720
K = np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 355.0], [0.0, 0.0, 1.0]])
NO_DIST = np.zeros(5)
# Typical cheap wide USB webcam (barrel distortion).
DIST = np.array([-0.32, 0.12, 0.0008, -0.0006, -0.02])

# Driver-station style camera: behind the blue wall, 2 m up, looking down-field.
DS_POSE = CameraPose.look_at([-0.5, 4.0, 2.0], [9.0, 4.2, 0.0])


def floor_grid(xs=np.linspace(2.0, 14.0, 7), ys=np.linspace(1.0, 7.0, 7)):
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def project(points, pose, dist):
    rvec, tvec = pose.to_rvec_tvec()
    px, _ = cv2.projectPoints(points.astype(np.float64), rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def in_image(px, margin=0):
    return (px[:, 0] >= margin) & (px[:, 0] < W - margin) & (px[:, 1] >= margin) & (px[:, 1] < H - margin)


# --------------------------------------------------------------------------- #
# Round trips against OpenCV's forward model
# --------------------------------------------------------------------------- #
def test_round_trip_no_distortion():
    pts = floor_grid()
    px = project(pts, DS_POSE, NO_DIST)
    keep = in_image(px)
    assert keep.sum() > 20
    xy, ok = project_pixels_to_floor(px[keep], K, NO_DIST, DS_POSE)
    assert ok.all()
    np.testing.assert_allclose(xy, pts[keep, :2], atol=1e-6)


def test_round_trip_with_distortion():
    pts = floor_grid()
    px = project(pts, DS_POSE, DIST)
    keep = in_image(px, margin=20)
    assert keep.sum() > 20
    xy, ok = project_pixels_to_floor(px[keep], K, DIST, DS_POSE)
    assert ok.all()
    # Undistortion is iterative; at 14 m a 0.01 px error is ~1 cm, so this is tight.
    np.testing.assert_allclose(xy, pts[keep, :2], atol=1e-3)


def test_distortion_matters():
    """Ignoring distortion must give a noticeably wrong answer away from the image centre."""
    pts = floor_grid()
    px = project(pts, DS_POSE, DIST)
    keep = in_image(px, margin=20)
    xy_wrong, _ = project_pixels_to_floor(px[keep], K, NO_DIST, DS_POSE)
    err = np.linalg.norm(xy_wrong - pts[keep, :2], axis=1)
    assert err.max() > 0.2


@pytest.mark.parametrize("roll_deg", [0.0, 3.0, -7.5])
def test_round_trip_rolled_camera(roll_deg):
    pose = CameraPose.look_at([16.9, 2.5, 1.6], [8.0, 4.0, 0.0], roll=np.radians(roll_deg))  # red side
    pts = floor_grid()
    px = project(pts, pose, DIST)
    keep = in_image(px, margin=20)
    assert keep.sum() > 10
    xy, ok = project_pixels_to_floor(px[keep], K, DIST, pose)
    assert ok.all()
    np.testing.assert_allclose(xy, pts[keep, :2], atol=1e-3)


def test_round_trip_through_solvepnp_pose():
    """A pose built from solvePnP's rvec/tvec projects consistently (catches transpose/sign slips)."""
    obj = np.array([[4.0, 3.5, 1.1], [4.0, 4.5, 1.1], [4.0, 4.5, 0.6], [4.0, 3.5, 0.6], [12.0, 1.0, 0.9], [12.0, 7.0, 0.9]])
    rvec, tvec = DS_POSE.to_rvec_tvec()
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, NO_DIST)
    ok, rv, tv = cv2.solvePnP(obj, img, K, NO_DIST, flags=cv2.SOLVEPNP_SQPNP)
    assert ok
    pose = CameraPose.from_rvec_tvec(rv, tv)
    np.testing.assert_allclose(pose.position, DS_POSE.position, atol=1e-6)
    np.testing.assert_allclose(pose.rotation, DS_POSE.rotation, atol=1e-6)

    target = np.array([[7.5, 2.0, 0.0]])
    xy, ok = project_pixels_to_floor(project(target, DS_POSE, NO_DIST), K, NO_DIST, pose)
    assert ok.all()
    np.testing.assert_allclose(xy[0], target[0, :2], atol=1e-6)


# --------------------------------------------------------------------------- #
# Analytic cases (no OpenCV forward model involved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("height,pitch_deg", [(1.0, 10.0), (2.0, 7.0), (2.5, 30.0)])
def test_principal_ray_hits_floor_at_h_over_tan_pitch(height, pitch_deg):
    # Camera at the origin looking along +X, tilted down by pitch.
    pose = CameraPose(rotation=rpy_to_rotation(0.0, np.radians(pitch_deg), 0.0) @ np.linalg.inv(
        np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])), position=np.array([0.0, 0.0, height]))
    xy, ok = project_pixels_to_floor(np.array([[K[0, 2], K[1, 2]]]), K, NO_DIST, pose)
    assert ok[0]
    np.testing.assert_allclose(xy[0], [height / np.tan(np.radians(pitch_deg)), 0.0], atol=1e-9)


def test_camera_straight_down():
    pose = CameraPose.look_at([3.0, 2.0, 5.0], [3.0, 2.0, 0.0])
    xy, ok = project_pixels_to_floor(np.array([[K[0, 2], K[1, 2]]]), K, NO_DIST, pose)
    assert ok[0]
    np.testing.assert_allclose(xy[0], [3.0, 2.0], atol=1e-9)
    # 90 px right of centre = 90/fx * 5 m sideways on the floor.
    xy, ok = project_pixels_to_floor(np.array([[K[0, 2] + 90.0, K[1, 2]]]), K, NO_DIST, pose)
    assert ok[0]
    assert np.isclose(np.linalg.norm(xy[0] - [3.0, 2.0]), 90.0 / K[0, 0] * 5.0)


def test_pixel_above_horizon_is_rejected():
    level = CameraPose.look_at([0.0, 4.0, 1.5], [10.0, 4.0, 1.5])  # optical axis horizontal
    px = np.array([[640.0, 100.0], [640.0, 355.0], [640.0, 700.0]])  # sky, horizon, floor
    xy, ok = project_pixels_to_floor(px, K, NO_DIST, level)
    assert ok.tolist() == [False, False, True]
    assert np.isnan(xy[0]).all() and np.isnan(xy[1]).all()
    assert np.isclose(xy[2, 0], 1.5 / ((700.0 - 355.0) / 905.0))


def test_camera_below_plane_and_behind():
    # Ray pointing down from below the plane never meets it in front of the camera.
    pts, ok = intersect_plane_z(np.array([0.0, 0.0, -1.0]), np.array([[0.0, 0.0, -1.0]]), 0.0)
    assert not ok[0] and np.isnan(pts[0]).all()
    # Horizontal ray never intersects.
    _, ok = intersect_plane_z(np.array([0.0, 0.0, 1.0]), np.array([[1.0, 0.0, 0.0]]), 0.0)
    assert not ok[0]


def test_nonzero_floor_plane():
    pose = CameraPose.look_at([0.0, 0.0, 2.0], [4.0, 0.0, 0.0])
    target = np.array([[3.0, 0.5, 0.3]])
    xy, ok = project_pixels_to_floor(project(target, pose, NO_DIST), K, NO_DIST, pose, floor_z=0.3)
    assert ok[0]
    np.testing.assert_allclose(xy[0], target[0, :2], atol=1e-9)


def test_ground_offset_moves_point_away_from_camera():
    pts = np.array([[6.0, 2.0, 0.0], [10.0, 6.5, 0.0]])
    px = project(pts, DS_POSE, NO_DIST)
    base, _ = project_pixels_to_floor(px, K, NO_DIST, DS_POSE)
    off, ok = project_pixels_to_floor(px, K, NO_DIST, DS_POSE, ground_offset_m=0.075)
    assert ok.all()
    np.testing.assert_allclose(np.linalg.norm(off - base, axis=1), 0.075, atol=1e-9)
    cam = DS_POSE.position[:2]
    assert np.all(np.linalg.norm(off - cam, axis=1) > np.linalg.norm(base - cam, axis=1))


def test_rays_are_unit_and_centre_ray_is_optical_axis():
    rays = pixel_rays(np.array([[K[0, 2], K[1, 2]], [0.0, 0.0], [W - 1.0, H - 1.0]]), K, DIST, DS_POSE)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
    np.testing.assert_allclose(rays[0], DS_POSE.rotation[:, 2], atol=1e-12)


def test_undistort_inverts_distort():
    norm = np.array([[0.0, 0.0], [0.3, -0.2], [-0.55, 0.3], [0.6, 0.35]])
    pts3 = np.column_stack([norm, np.ones(len(norm))])
    px, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, DIST)
    np.testing.assert_allclose(undistort_to_normalized(px.reshape(-1, 2), K, DIST), norm, atol=1e-7)


def test_empty_input():
    xy, ok = project_pixels_to_floor(np.zeros((0, 2)), K, DIST, DS_POSE)
    assert xy.shape == (0, 2) and ok.shape == (0,)


# --------------------------------------------------------------------------- #
# Field bounds
# --------------------------------------------------------------------------- #
def test_field_bounds():
    xy = np.array([[0.0, 0.0], [16.5, 8.0], [-0.01, 4.0], [8.0, 8.1], [np.nan, np.nan], [17.0, 4.0]])
    assert in_field_bounds(xy, 16.541, 8.069).tolist() == [True, True, False, False, False, False]
    assert in_field_bounds(xy, 16.541, 8.069, margin=0.1).tolist() == [True, True, True, True, False, False]
    assert in_field_bounds(xy, 16.541, 8.069, margin=-0.1).tolist() == [False, False, False, False, False, False]


# --------------------------------------------------------------------------- #
# Published pose conventions
# --------------------------------------------------------------------------- #
def test_rpy_round_trip():
    for r, p, y in [(0.1, -0.2, 0.3), (0.0, 0.0, np.pi - 0.01), (-0.5, 0.7, -2.0)]:
        np.testing.assert_allclose(rotation_to_rpy(rpy_to_rotation(r, p, y)), [r, p, y], atol=1e-12)


def test_camera_rpy_wpilib_convention():
    # Looking down-field (+X) and tilted down: yaw 0, positive pitch, no roll.
    pose = CameraPose.look_at([0.0, 0.0, 2.0], [10.0, 0.0, 0.0])
    roll, pitch, yaw = pose.rpy()
    assert np.isclose(roll, 0.0, atol=1e-12)
    assert np.isclose(pitch, np.arctan2(2.0, 10.0))
    assert np.isclose(yaw, 0.0, atol=1e-12)
    # Looking towards +Y (left from the blue DS): yaw +90 deg.
    _, _, yaw = CameraPose.look_at([8.0, 0.0, 1.0], [8.0, 5.0, 1.0]).rpy()
    assert np.isclose(yaw, np.pi / 2)
    # as_list order: x, y, z, roll, pitch, yaw
    assert CameraPose.look_at([1.0, 2.0, 3.0], [5.0, 2.0, 3.0]).as_list()[:3] == [1.0, 2.0, 3.0]


def test_project_field_points_marks_points_behind_camera():
    pts = np.array([[5.0, 4.0, 0.0], [-3.0, 4.0, 0.0]])
    _, front = project_field_points(pts, K, NO_DIST, DS_POSE)
    assert front.tolist() == [True, False]
