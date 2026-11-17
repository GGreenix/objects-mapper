# objects-mapper — FRC driver-station field mapper

A fixed USB camera at the driver station looks across the field. This app works out exactly where that camera is from the field AprilTags, detects game pieces with YOLO, projects them onto the floor, and publishes their field X/Y to the robot over NetworkTables.

```
USB camera ──► capture ──┬──► AprilTags (every 6 s) ──► solvePnP ──► pose filter ──► camera pose
                         │                                                              │
                         └──► YOLO (every frame) ──► box bottom-centre ──► undistort ──►┴► ray ∩ floor (z=0)
                                                                                              │
                              NetworkTables /FieldMapper ◄── tracker ◄── field-bounds filter ◄┘
```

Stack: Python 3.11, OpenCV, robotpy-apriltag (bundled WPILib 2026 *REBUILT* layouts), Ultralytics YOLO on CUDA, pyntcore.

---

## Install

Use Python 3.11 in a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)

# 1. PyTorch with CUDA (RTX 4050). Get the exact command from https://pytorch.org/get-started/locally/
#    for your CUDA version. Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements-dev.txt
```

Check that CUDA works: `python -c "import torch; print(torch.cuda.is_available())"`. If it doesn't, the mapper still runs, on CPU, and logs a warning.

`opencv-python` (not `-headless`) is required for the debug windows. Don't install both builds at once.

## 1. Calibrate the camera

Camera intrinsics and distortion go in `calib.yaml`. **Calibrate at the resolution you will run at, with autofocus off.** Calibration is only valid for that focus setting.

1. Print a checkerboard and glue it to something flat. The default is **9×6 inner corners** (10×7 squares) with 25 mm squares. Set `calibration:` in `config.yaml` or pass `--cols/--rows/--square`. Measure the printed square size.
2. Run:

   ```bash
   python calibrate.py
   ```

   | key | action |
   |---|---|
   | `SPACE` | capture the current view |
   | `a` | toggle auto-capture (grabs a view when the board is still and in a new position) |
   | `u` | undo last capture |
   | `c` | calibrate and save (needs `min_frames`, default 15) |
   | `q` / `Esc` | quit without saving |

   Cover the **whole image**, corners and edges especially, at different distances and tilts. The green dots show coverage so far.
3. Check the printed RMS error. Under ~0.5 px is good; over 1 px means something is wrong (motion blur, a warped board, or the wrong board size).

Offline alternatives: `python calibrate.py --images "calib_images/*.png"` or `--video calib.mp4 --every 10`. Use `--save-dir calib_images` in live mode to keep the captured frames. For very wide lenses, add `--rational`.

If you change capture resolution later with the same aspect ratio, the intrinsics are scaled automatically, with a warning. Recalibrating is better.

## 2. Configure

Edit `config.yaml`. Every key has a default (`field_mapper/config.py`), and unknown keys are rejected so typos don't go unnoticed. The ones you must set:

| key | what |
|---|---|
| `team_number` | used to find the robot (plus the Driver Station, if it's running on this laptop) |
| `camera.index`, `width`, `height` | which USB camera and its mode |
| `field.layout` | `k2026RebuiltWelded` or `k2026RebuiltAndyMark`. **Check which field your event uses.** Can also be a path to a WPILib layout JSON |
| `detector.model_path` | your trained YOLO `.pt` file |
| `detector.classes` | class names or ids to keep (empty = all) |

Useful extras: `projection.ground_offset_m` (set it to the game piece radius to report piece centres instead of near edges), `networktables.server: "127.0.0.1"` for simulation, `camera.properties` to lock focus and exposure.

## 3. Run

```bash
python main.py                         # live camera, debug windows per config
python main.py --headless              # no windows (competition)
python main.py --replay match.mp4      # run on a recording at real-time speed
python main.py --replay match.mp4 --speed 0 --loop --no-nt
```

| flag | |
|---|---|
| `--config PATH` | config file (default `config.yaml`) |
| `--replay VIDEO` | read a video file instead of the camera. Time-based behaviour (re-localising every 6 s) follows the video clock |
| `--speed X` | replay speed, `1` = real time, `0` = as fast as possible |
| `--loop` | loop the replay |
| `--no-nt` | don't publish to NetworkTables |
| `--debug` / `--headless` | force the debug windows on/off |
| `--calib`, `--model` | override the calibration / model paths |
| `-v` | debug logging |

**Debug windows.** *Camera*: detected tags are green when used in the last accepted solve and orange otherwise. Yellow outlines and the grey perimeter show where the layout says the tags and field edges should appear given the current pose; if they line up with the real ones, the pose is good. Detections are labelled with their field coordinates (green: on field, orange: off field, red: no pose). *Field*: top-down view with the tags, the camera and its floor footprint, raw projections (grey dots) and tracked pieces with IDs (a hollow circle means not seen this frame).

Keys: `q`/`Esc` quit, `r` re-localise now, `SPACE` pause (replay).

## NetworkTables output

Table `/FieldMapper`, published every processed frame:

| topic | type | meaning |
|---|---|---|
| `x` | `double[]` | field X of each tracked piece, metres |
| `y` | `double[]` | field Y of each tracked piece, metres (same order as `x`) |
| `timestamp` | `double` | frame capture time, seconds, **on the NT server clock**, so on the robot it is directly comparable with `Timer.getFPGATimestamp()` |
| `cameraPose` | `double[6]` | camera `x, y, z` (m), `roll, pitch, yaw` (rad) |
| `poseValid` | `boolean` | `x`/`y` are only meaningful when true |

Coordinates use the WPILib field frame with the **blue-alliance origin**, as robot code expects: +X from the blue wall towards red, +Y to the left as seen from the blue driver station, +Z up. Set `field.origin: red` if you want red-origin coordinates. `cameraPose` angles follow WPILib `Rotation3d(roll, pitch, yaw)` for the camera body (X out of the lens, Y left, Z up), so positive pitch means tilted down. Pass them straight to `new Pose3d(x, y, z, new Rotation3d(roll, pitch, yaw))`.

Robot-side example (Java):

```java
NetworkTable t = NetworkTableInstance.getDefault().getTable("FieldMapper");
DoubleArraySubscriber xSub = t.getDoubleArrayTopic("x").subscribe(new double[0]);
DoubleArraySubscriber ySub = t.getDoubleArrayTopic("y").subscribe(new double[0]);
DoubleSubscriber tsSub = t.getDoubleTopic("timestamp").subscribe(0);
BooleanSubscriber validSub = t.getBooleanTopic("poseValid").subscribe(false);

double[] xs = xSub.get(), ys = ySub.get();
if (validSub.get() && xs.length == ys.length) {   // x and y arrive as separate updates
  double ageSeconds = Timer.getFPGATimestamp() - tsSub.get();
  for (int i = 0; i < xs.length; i++) {
    Translation2d piece = new Translation2d(xs[i], ys[i]);
    // ...
  }
}
```

## How it works

**Localisation** (`field_mapper/localizer.py`)
- Every visible 36h11 tag's 4 corners go into **one** solvePnP against their 3D field positions from the layout (SQPnP, then Levenberg–Marquardt refinement).
- A single tag uses IPPE. It gives the two classic planar solutions: any solution that puts the camera under the floor is dropped, then the one nearest the currently held pose wins. If there is no held pose and the two are too similar (`max_single_tag_ambiguity`), the solve is refused.
- Re-solves every `interval_s` (6 s). With no pose yet, or while confirming a possible camera move, it retries every `retry_interval_s` instead.
- A new solve is **rejected** if its RMS reprojection error is above `max_reprojection_error_px`, or if it is more than `max_jump_m` / `max_jump_deg` from the held pose. Accepted solves are **blended** in (`smoothing_alpha`; slerp for rotation) rather than snapped.
- If the camera really was bumped, `jump_confirmations` consecutive solves that agree with *each other* re-anchor the pose. Inconsistent outliers never do. Set `jump_confirmations: 0` to disable.
- `max_pose_age_s` can invalidate a pose that hasn't been re-confirmed for a while (off by default, since robots often block the tags).

**Projection** (`field_mapper/geometry.py`)
1. Take the bottom-centre pixel of each detection box.
2. Undistort it to normalised camera coordinates (iterating to convergence).
3. Rotate the ray into the field frame and intersect it with z = `floor_z_m`.
4. Rays at or above the horizon are discarded, as are points outside the field (± `bounds_margin_m`).

**Tracking** (`field_mapper/tracker.py`): greedy nearest-neighbour in field metres, per class, with a gate of `max_match_distance_m`.
- A piece must be seen `min_hits` times before it is published.
- It stays published, at its last position, for up to `max_missed_frames` frames without a match, so it doesn't flicker.
- Positions are exponentially smoothed.

## Accuracy: what to expect

This camera views the floor at a very shallow angle, and that dominates everything. For a camera at height *h*, a piece at ground distance *d* and focal length *f* (pixels):

- **range error ≈ (h² + d²) / (h·f)** metres per pixel of error at the box's bottom edge
- **sideways error ≈ d / f** metres per pixel

| camera height | resolution (70° HFOV) | range error per px at 5 m / 10 m / 15 m |
|---|---|---|
| 2.0 m | 1280×720 (f≈914) | 1.6 / 5.7 / 12.5 cm |
| 2.0 m | 1920×1080 (f≈1371) | 1.0 / 3.8 / 8.4 cm |
| 1.5 m | 1920×1080 | 1.3 / 5.0 / 11.0 cm |

Practical consequences:
- **Mount the camera as high as you're allowed, and use 1080p or more.**
- **Camera pitch accuracy matters most.** 0.1° of pitch error is about 9 cm of range error at 10 m (h = 2 m). Multi-tag solves with tags spread across the image are far better than one tag, which is why `min_tags` defaults to 2. In synthetic tests, a single tag 4.6 m away (~33 px) with 0.3 px corner noise gave 0.3–0.7 m position error.
- **Tags must be big enough to decode.** A 6.5 in tag is f·0.165/d pixels wide: at 720p that's ~30 px at 5 m and ~9 px across the full field. Keep `quad_decimate: 1` and `min_cluster_pixels` low. WPILib's default of 300 silently drops every tag smaller than ~75 px, which is all of them from the driver station.
- Pieces **not on the floor** (in a robot, on a ledge, stacked) project to the wrong place, along the line of sight beyond their true position. Pieces hidden behind robots aren't seen at all.
- The box bottom is the piece's near edge. `projection.ground_offset_m` (≈ piece radius) shifts it to the centre.

## Tests

```bash
pytest
```

- `tests/test_projection.py` covers the floor projection maths:
  - round trips through OpenCV's forward model, with and without lens distortion and with camera roll;
  - analytic cases (principal ray at h/tan(pitch), straight-down camera);
  - rejection above the horizon, non-zero floor planes, ground offset, field bounds;
  - the solvePnP pose conventions and the published roll/pitch/yaw.
- `tests/test_localizer.py`:
  - multi-tag and single-tag PnP against the real 2026 layout;
  - the pose filter (blend, reject, re-anchor, expiry);
  - an end-to-end test that renders the field tags, detects them with robotpy-apriltag and recovers the camera pose.
- `tests/test_tracker.py` covers the tracker.

## Layout

```
main.py                 entry point (python main.py ...)
calibrate.py            checkerboard calibration -> calib.yaml
config.yaml             all settings
field_mapper/
  app.py                CLI + main loop
  capture.py            threaded live camera (always newest frame), video replay
  calibration.py        calib.yaml load/save, resolution scaling
  config.py             typed config with defaults
  field.py              field layout -> tag corner positions
  localizer.py          AprilTag detection, PnP, pose filter, scheduler
  detector.py           Ultralytics YOLO wrapper
  geometry.py           pose maths, undistort, ray/floor intersection
  tracker.py            nearest-neighbour tracker
  publisher.py          NetworkTables output
  debug_view.py         camera overlay + top-down field view
tests/
```
