"""Browser calibration flow (CalibrationSession) on rendered, distorted checkerboards."""

import time

import cv2
import numpy as np

from field_mapper.calib_core import CalibrationSession, printable_board
from field_mapper.calibration import load_intrinsics

from .synthetic import render_checkerboard_views

W, H = 960, 540
K = np.array([[720.0, 0, 482.0], [0, 718.0, 268.0], [0, 0, 1]])
D = np.array([-0.22, 0.07, 0.0004, -0.0003, -0.01])


def feed(session, img, frames=3):
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for _ in range(frames):
        session.on_frame(bgr)
        session._last_quick_time = 0.0  # don't wait for the ~8 Hz preview throttle in tests


def test_session_captures_and_calibrates(tmp_path):
    out = tmp_path / "calib.yaml"
    session = CalibrationSession(9, 6, 0.025, min_views=12, output=out)
    views = render_checkerboard_views(K, D, (W, H), n_views=16)

    feed(session, np.full((H, W), 128, np.uint8))
    session.request_capture()
    feed(session, np.full((H, W), 128, np.uint8), 1)
    assert session.state()["views"] == 0 and "No board" in session.state()["message"]

    for v in views:
        feed(session, v, 1)
        session.request_capture()
        feed(session, v, 1)
    st = session.state()
    assert st["views"] >= 12, st
    assert session.start_solve()
    deadline = time.time() + 60
    while session.solving and time.time() < deadline:
        time.sleep(0.05)
    st = session.state()
    assert st["saved"], st["message"]
    assert st["result"]["rms_px"] < 0.3
    intr = load_intrinsics(out)
    assert (intr.width, intr.height) == (W, H)
    np.testing.assert_allclose(intr.camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]], [720, 718, 482, 268], atol=3.0)
    np.testing.assert_allclose(intr.dist_coeffs[:2], D[:2], atol=0.03)


def test_auto_capture_needs_a_still_new_view(tmp_path):
    session = CalibrationSession(9, 6, 0.025, min_views=5, output=tmp_path / "c.yaml")
    session.set_auto(True)
    views = render_checkerboard_views(K, D, (W, H), n_views=2, seed=7)
    feed(session, views[0], 3)
    assert session.state()["views"] == 1
    session._last_capture = 0.0
    feed(session, views[0], 3)  # same view again: not new enough
    assert session.state()["views"] == 1
    session._last_capture = 0.0
    feed(session, views[1], 3)
    assert session.state()["views"] == 2
    session.undo()
    assert session.state()["views"] == 1
    session.clear()
    assert session.state()["views"] == 0


def test_cannot_solve_without_enough_views(tmp_path):
    session = CalibrationSession(9, 6, 0.025, min_views=15, output=tmp_path / "c.yaml")
    assert session.start_solve() is False


def test_printable_board_scale():
    img = printable_board(9, 6, 25.0, dpi=300)
    sq = round(25.0 / 25.4 * 300)
    assert img.shape[1] == 12 * sq  # 10 squares + 2 margins
    ok, _ = cv2.findChessboardCorners(cv2.resize(img, None, fx=0.25, fy=0.25), (9, 6))
    assert ok
