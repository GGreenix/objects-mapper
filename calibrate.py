#!/usr/bin/env python3
"""Checkerboard camera calibration -> calib.yaml.

Live (uses camera.index / width / height from config.yaml):

    python calibrate.py

    SPACE  capture the current view      a  toggle auto-capture
    u      undo last capture             c  calibrate and save
    q/ESC  quit without saving

Offline, from saved images or a video:

    python calibrate.py --images "calib_images/*.png"
    python calibrate.py --video calib.mp4 --every 10

Board size is given in *inner* corners (a board with 10x7 squares has 9x6
inner corners). Calibrate at the same resolution you run the mapper at, with
autofocus off, and move the board over the whole image - corners and edges
matter most for the distortion model.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from field_mapper.calibration import CameraIntrinsics, save_intrinsics
from field_mapper.capture import open_camera
from field_mapper.config import Config, load_config

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Accurate full-resolution corner detection. Returns (N, 2) or None."""
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners.reshape(-1, 2).astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners.reshape(-1, 2).astype(np.float32)


def quick_corners(gray: np.ndarray, pattern: tuple[int, int], max_width: int = 640) -> np.ndarray | None:
    """Fast, approximate detection on a downscaled image (for the live preview)."""
    scale = min(1.0, max_width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    return corners.reshape(-1, 2) / scale if ok else None


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def calibrate(
    views: list[np.ndarray], pattern: tuple[int, int], square: float, size: tuple[int, int], rational: bool = False
) -> tuple[CameraIntrinsics, list[float], int]:
    """Runs cv2.calibrateCamera; drops clearly bad views (> 3x median error) once and re-solves."""
    obj = board_points(pattern, square)
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0

    def solve(vs):
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera([obj] * len(vs), [v.reshape(-1, 1, 2) for v in vs], size, None, None, flags=flags)
        errs = []
        for v, r, t in zip(vs, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, r, t, k, d)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v) ** 2, axis=1)))))
        return rms, k, d, errs

    rms, k, d, errs = solve(views)
    med = float(np.median(errs))
    keep = [v for v, e in zip(views, errs) if e <= max(3 * med, 0.5)]
    dropped = len(views) - len(keep)
    if dropped and len(keep) >= 5:
        print(f"dropping {dropped} view(s) with error > {max(3 * med, 0.5):.2f}px and re-solving")
        rms, k, d, errs = solve(keep)
    else:
        dropped = 0
    return CameraIntrinsics(k, d.reshape(-1), size[0], size[1], float(rms)), errs, dropped


def report_and_save(intr: CameraIntrinsics, errs, n_views: int, dropped: int, out: Path, pattern, square) -> None:
    k = intr.camera_matrix
    hfov = np.degrees(2 * np.arctan(intr.width / (2 * k[0, 0])))
    vfov = np.degrees(2 * np.arctan(intr.height / (2 * k[1, 1])))
    print(f"\nviews used: {n_views - dropped} (dropped {dropped})")
    print(f"RMS reprojection error: {intr.rms_px:.3f} px  (worst view {max(errs):.3f} px)")
    print(f"fx={k[0, 0]:.1f} fy={k[1, 1]:.1f} cx={k[0, 2]:.1f} cy={k[1, 2]:.1f}   FOV {hfov:.1f} x {vfov:.1f} deg")
    print("dist:", np.array2string(intr.dist_coeffs, precision=5))
    if intr.rms_px > 1.0:
        print("WARNING: RMS > 1 px - check focus/motion blur, board flatness and the board size arguments")
    save_intrinsics(
        out, intr,
        extra={"num_views": n_views - dropped, "board": {"cols": pattern[0], "rows": pattern[1], "square_size_m": square}},
    )
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
def collect_offline(args, pattern) -> tuple[list[np.ndarray], tuple[int, int] | None]:
    views, size = [], None
    if args.images:
        files = sorted(glob.glob(args.images))
        if not files:
            sys.exit(f"no images match {args.images}")
        frames = ((f, cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files)
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"could not open {args.video}")

        def gen():
            i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                if i % args.every == 0:
                    yield f"frame {i}", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                i += 1

        frames = gen()
    for name, gray in frames:
        if gray is None:
            print(f"{name}: unreadable")
            continue
        s = (gray.shape[1], gray.shape[0])
        if size is None:
            size = s
        elif s != size:
            print(f"{name}: size {s} != {size}, skipped")
            continue
        c = find_corners(gray, pattern)
        print(f"{name}: {'ok' if c is not None else 'no board'}")
        if c is not None:
            views.append(c)
    return views, size


def is_new_view(c: np.ndarray, views: list[np.ndarray], diag: float, min_diff: float = 0.05) -> bool:
    if not views:
        return True
    diffs = [float(np.mean(np.linalg.norm(c - v, axis=1))) / diag for v in views]
    return min(diffs) > min_diff


def collect_live(cfg: Config, args, pattern) -> tuple[list[np.ndarray], tuple[int, int] | None]:
    if args.camera is not None:
        cfg.camera.index = args.camera
    cap = open_camera(cfg.camera)
    views: list[np.ndarray] = []
    quick_views: list[np.ndarray] = []  # preview-detector corners, for the "is this view new?" check
    size = None
    auto = False
    last_capture = 0.0
    prev_quick = None
    coverage = None
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    calibrate_now = False
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            if coverage is None:
                coverage = np.zeros_like(img)
            diag = float(np.hypot(*size))
            quick = quick_corners(gray, pattern)
            still = (
                quick is not None and prev_quick is not None and float(np.mean(np.linalg.norm(quick - prev_quick, axis=1))) < 0.003 * diag
            )
            prev_quick = quick

            key = cv2.waitKey(1) & 0xFF
            want = key == ord(" ") or (
                auto and still and time.monotonic() - last_capture > 1.0 and is_new_view(quick, quick_views, diag)
            )
            if want and quick is not None:
                c = find_corners(gray, pattern)
                if c is not None:
                    views.append(c)
                    quick_views.append(quick)
                    last_capture = time.monotonic()
                    for p in c:
                        cv2.circle(coverage, (int(p[0]), int(p[1])), 3, (0, 200, 0), -1)
                    if save_dir:
                        cv2.imwrite(str(save_dir / f"calib_{len(views):03d}.png"), img)
                    print(f"captured view {len(views)}")
            elif key == ord(" "):
                print("no board found")
            if key == ord("a"):
                auto = not auto
                print(f"auto-capture {'on' if auto else 'off'}")
            if key == ord("u") and views:
                views.pop()
                quick_views.pop()
                coverage[:] = 0
                for v in views:
                    for p in v:
                        cv2.circle(coverage, (int(p[0]), int(p[1])), 3, (0, 200, 0), -1)
                print(f"removed last view ({len(views)} left)")
            if key == ord("c"):
                if len(views) < args.min_frames:
                    print(f"need at least {args.min_frames} views (have {len(views)})")
                else:
                    calibrate_now = True
                    break
            if key in (ord("q"), 27):
                break

            vis = cv2.addWeighted(img, 1.0, coverage, 0.6, 0)
            if quick is not None:
                cv2.drawChessboardCorners(vis, pattern, quick.reshape(-1, 1, 2).astype(np.float32), True)
            lines = [
                f"views {len(views)}/{args.min_frames}   auto {'ON' if auto else 'off'}   board {'still' if still else ('seen' if quick is not None else '-')}",
                "SPACE capture  a auto  u undo  c calibrate+save  q quit",
            ]
            for i, line in enumerate(lines):
                cv2.putText(vis, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("calibrate", vis)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return (views if calibrate_now else []), size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Checkerboard camera calibration")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--images", help="glob of calibration images (offline mode)")
    p.add_argument("--video", help="calibration video (offline mode)")
    p.add_argument("--every", type=int, default=10, help="with --video: use every Nth frame")
    p.add_argument("--camera", type=int, help="override camera.index")
    p.add_argument("--cols", type=int, help="inner corners per row")
    p.add_argument("--rows", type=int, help="inner corners per column")
    p.add_argument("--square", type=float, help="square size in metres")
    p.add_argument("--min-frames", type=int, help="minimum views before calibrating")
    p.add_argument("--rational", action="store_true", help="8-coefficient rational model (very wide lenses)")
    p.add_argument("--save-dir", help="live mode: also save captured images here")
    p.add_argument("-o", "--output", help="output file (default: camera.calibration_file)")
    args = p.parse_args(argv)

    cfg = load_config(args.config) if Path(args.config).exists() else Config()
    c = cfg.calibration
    pattern = (args.cols or c.board_cols, args.rows or c.board_rows)
    square = args.square or c.square_size_m
    args.min_frames = args.min_frames or c.min_frames
    out = Path(args.output) if args.output else cfg.resolve_path(cfg.camera.calibration_file)
    print(f"board {pattern[0]}x{pattern[1]} inner corners, {square * 1000:.1f} mm squares -> {out}")

    if args.images or args.video:
        views, size = collect_offline(args, pattern)
    else:
        views, size = collect_live(cfg, args, pattern)
        if not views:
            print("not calibrated")
            return 1

    if len(views) < args.min_frames:
        print(f"only {len(views)} usable views, need {args.min_frames}")
        return 1
    print(f"calibrating with {len(views)} views at {size[0]}x{size[1]} ...")
    intr, errs, dropped = calibrate(views, pattern, square, size, args.rational)
    report_and_save(intr, errs, len(views), dropped, out, pattern, square)
    return 0


if __name__ == "__main__":
    sys.exit(main())
