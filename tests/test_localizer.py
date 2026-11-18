"""Camera localisation: PnP over the real 2026 layout, pose filtering, and an
end-to-end check with rendered tags through the real AprilTag detector."""

import numpy as np
import pytest

from field_mapper.config import AprilTagConfig, LocalizationConfig
from field_mapper.field import FieldLayout, tag_corners_local
from field_mapper.geometry import CameraPose, project_field_points
from field_mapper.localizer import Localizer, PoseFilter, PoseSolution, TagObservation, solve_camera_pose

from .synthetic import make_intrinsics, render_tags

K = make_intrinsics()
D0 = np.zeros(5)
TRUE_POSE = CameraPose.look_at([-0.6, 4.0, 1.9], [10.0, 4.0, 0.0])


@pytest.fixture(scope="module")
def field():
    return FieldLayout.load("k2026RebuiltWelded", 0.1651)


def observe(field, pose, ids, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for tid in ids:
        px, front = project_field_points(field.corners(tid), K, D0, pose)
        assert front.all()
        out.append(TagObservation(tid, px + rng.normal(0, noise_px, px.shape)))
    return out


def solution(pose, err=0.3, ids=(1, 2)):
    return PoseSolution(pose, err, list(ids))


# --------------------------------------------------------------------------- #
def test_layout_loads_2026(field):
    assert len(field.tags) == 32
    assert 16.4 < field.length < 16.6 and 8.0 < field.width < 8.1


def test_tag_corner_order_matches_detector_convention():
    # Facing the tag (looking along -X), +Y is on the viewer's right.
    bl, br, tr, tl = tag_corners_local(1.0)
    assert bl[1] < br[1] and bl[2] < tl[2] and tr[1] > tl[1] and tr[2] > br[2]


def test_multi_tag_solve_exact(field):
    obs = observe(field, TRUE_POSE, [23, 25, 26, 28])
    sol = solve_camera_pose(obs, field, K, D0)
    assert sol is not None and sol.reprojection_error_px < 1e-6
    np.testing.assert_allclose(sol.pose.position, TRUE_POSE.position, atol=1e-6)


def test_multi_tag_solve_with_noise(field):
    obs = observe(field, TRUE_POSE, [1, 3, 4, 6, 23, 25, 26, 28], noise_px=0.5)
    sol = solve_camera_pose(obs, field, K, D0)
    assert sol.reprojection_error_px < 1.0
    dt, dr = sol.pose.distance_to(TRUE_POSE)
    assert dt < 0.05 and np.degrees(dr) < 0.3


def test_unknown_tags_are_ignored(field):
    obs = observe(field, TRUE_POSE, [25, 26]) + [TagObservation(99, np.zeros((4, 2)))]
    sol = solve_camera_pose(obs, field, K, D0)
    assert sorted(sol.tag_ids) == [25, 26]
    assert solve_camera_pose([TagObservation(99, np.zeros((4, 2)))], field, K, D0) is None


def test_min_tags(field):
    obs = observe(field, TRUE_POSE, [25])
    assert solve_camera_pose(obs, field, K, D0, min_tags=2) is None


def test_single_tag_noise_free_picks_true_branch(field):
    sol = solve_camera_pose(observe(field, TRUE_POSE, [25]), field, K, D0)
    assert sol.pose.distance_to(TRUE_POSE)[0] < 1e-4
    assert sol.ambiguity < 1e-3


def test_single_tag_uses_prior_to_disambiguate(field):
    # With 0.3 px corner noise on a ~33 px tag 4.6 m away, the *wrong* IPPE branch
    # has the lower reprojection error for this seed: classic planar ambiguity.
    obs = observe(field, TRUE_POSE, [25], noise_px=0.3, seed=0)
    blind = solve_camera_pose(obs, field, K, D0)
    assert blind.pose.distance_to(TRUE_POSE)[0] > 1.0
    assert blind.ambiguity > 0.5  # ...which the filter refuses as a first pose
    assert not PoseFilter(cfg()).update(blind, 0.0)[0]
    # A held pose picks the right branch.
    guided = solve_camera_pose(obs, field, K, D0, prior=TRUE_POSE)
    assert guided.pose.distance_to(TRUE_POSE)[0] < 0.2


# --------------------------------------------------------------------------- #
def cfg(**kw):
    base = dict(max_reprojection_error_px=2.0, max_jump_m=0.25, max_jump_deg=5.0, smoothing_alpha=0.5, jump_confirmations=3)
    base.update(kw)
    return LocalizationConfig(**base)


def shifted(pose, dx):
    return CameraPose(pose.rotation, pose.position + np.array([dx, 0.0, 0.0]))


def test_filter_first_pose_snaps_then_blends():
    f = PoseFilter(cfg())
    assert not f.is_valid(0.0)
    assert f.update(solution(TRUE_POSE), 0.0)[0]
    np.testing.assert_allclose(f.pose.position, TRUE_POSE.position)
    assert f.update(solution(shifted(TRUE_POSE, 0.1)), 6.0)[0]
    np.testing.assert_allclose(f.pose.position[0], TRUE_POSE.position[0] + 0.05)


def test_filter_rejects_high_reprojection_error():
    f = PoseFilter(cfg())
    ok, reason = f.update(solution(TRUE_POSE, err=5.0), 0.0)
    assert not ok and "reprojection" in reason and f.pose is None


def test_filter_rejects_ambiguous_first_single_tag():
    f = PoseFilter(cfg(max_single_tag_ambiguity=0.2))
    ok, _ = f.update(PoseSolution(TRUE_POSE, 0.1, [25], ambiguity=0.8), 0.0)
    assert not ok


def test_filter_rejects_jump_then_reanchors_after_consistent_solves():
    f = PoseFilter(cfg(jump_confirmations=3))
    f.update(solution(TRUE_POSE), 0.0)
    moved = shifted(TRUE_POSE, 1.0)
    assert not f.update(solution(moved), 1.0)[0]
    assert not f.update(solution(TRUE_POSE.blend(moved, 1.0)), 2.0)[0]
    np.testing.assert_allclose(f.pose.position, TRUE_POSE.position)  # held pose untouched
    ok, reason = f.update(solution(moved), 3.0)
    assert ok and "moved" in reason
    np.testing.assert_allclose(f.pose.position, moved.position)


def test_filter_inconsistent_outliers_never_reanchor():
    f = PoseFilter(cfg(jump_confirmations=2))
    f.update(solution(TRUE_POSE), 0.0)
    for i, dx in enumerate([1.0, -1.0, 1.0, -1.0]):
        assert not f.update(solution(shifted(TRUE_POSE, dx)), float(i + 1))[0]
    np.testing.assert_allclose(f.pose.position, TRUE_POSE.position)


def test_filter_jump_rejection_can_be_permanent():
    f = PoseFilter(cfg(jump_confirmations=0))
    f.update(solution(TRUE_POSE), 0.0)
    for i in range(5):
        assert not f.update(solution(shifted(TRUE_POSE, 1.0)), float(i + 1))[0]


def test_filter_pose_expiry():
    f = PoseFilter(cfg(max_pose_age_s=10.0))
    f.update(solution(TRUE_POSE), 0.0)
    assert f.is_valid(9.0) and not f.is_valid(11.0)


# --------------------------------------------------------------------------- #
def test_end_to_end_rendered_tags(field):
    """Render the real field tags, detect with robotpy-apriltag, solve, and check the schedule."""
    img, rendered = render_tags(field, TRUE_POSE, K)
    assert len(rendered) >= 4
    loc = Localizer(LocalizationConfig(interval_s=6.0, retry_interval_s=0.5), AprilTagConfig(), field, K, D0)
    assert loc.due(0.0)
    res = loc.process(img, 0.0)
    assert res.accepted, res.reason
    assert len(res.solution.tag_ids) >= 4
    dt, dr = loc.pose.distance_to(TRUE_POSE)
    assert dt < 0.05 and np.degrees(dr) < 0.3
    assert loc.pose_valid(0.0)
    # Next solve is a full interval away once a pose is held.
    assert not loc.due(5.9) and loc.due(6.0)


def test_localizer_retries_quickly_without_pose(field):
    loc = Localizer(LocalizationConfig(interval_s=6.0, retry_interval_s=0.5), AprilTagConfig(), field, K, D0)
    blank = np.full((720, 1280, 3), 90, np.uint8)
    res = loc.process(blank, 0.0)
    assert not res.accepted and loc.pose is None
    assert loc.due(0.5) and not loc.due(0.4)
