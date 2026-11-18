# objects-mapper — FRC driver-station field mapper

A fixed USB camera at the driver station looks across the field. Field Mapper:
1. works out exactly where the camera is from the field AprilTags;
2. finds game pieces with YOLO;
3. puts them on the field map;
4. sends their X/Y positions to the robot over NetworkTables.

You run and set it up entirely from a web page in your browser.

```
USB camera ──► AprilTags (every 6 s) ──► camera pose ─┐
           └─► YOLO (every frame) ──► box bottom ─────┴─► ray ∩ floor ──► tracker ──► NetworkTables /FieldMapper
                                                                                  └─► browser dashboard
```

## Quick start

**Windows:** double-click **`start.bat`**.
**Linux / macOS:** run `./start.sh`.

- **The first time** it sets everything up, which takes 5–10 minutes and needs internet. Do this before the event.
  - If Python 3.11 is missing on Windows, it offers to install it.
  - It installs the GPU (CUDA) build of PyTorch that matches your NVIDIA driver, or a CPU build if there is no NVIDIA GPU.
  - It installs everything else.
- **Every time** your browser opens the dashboard at **http://localhost:5805**. The **Finish setup** checklist walks you through what's left:
  1. **Team number**, so the app can find your robot.
  2. **Camera**: *Settings → Find cameras* shows a preview of each connected camera. Click yours.
  3. **Calibrate**: print the checkerboard from the Calibrate tab and capture about 15 views. Takes about 3 minutes, once per camera.
  4. **Detection model**: upload your trained YOLO `.pt` file in *Settings*.
  5. **Aim the camera** so at least two AprilTags are visible.

After that, just double-click `start.bat`: it starts in seconds and works offline. Closing the console window stops it. Double-clicking again while it's running just reopens the page.

## The dashboard

**Status chips** (top right) show at a glance whether the camera, calibration, model, camera pose and robot connection are OK. Click a red one to go where it's fixed.

**Live**
- **Camera:** the camera image with overlays.
  - Detected tags are green when used for the pose.
  - Yellow outlines and the grey field edge show where the tags and field *should* be. If they line up with the real ones, the pose is good.
  - Each detection is labelled with its field position.
- **Field:** a top-down map showing the tags, the camera and what it can see, and the tracked pieces with IDs. Hover for coordinates. *View from red* flips it.
- **Game pieces:** the same list the robot receives.
- **Camera pose:** position, angles, and the result of the last check. *Re-localise now* forces a new check.
- **Controls:**
  - detection confidence slider (applies instantly);
  - *Reset tracker*;
  - **● Record**, which saves the raw camera feed to `recordings/`;
  - frame rate and timing.
- **Log:** what the app is doing, including why a pose check was rejected.

**Calibrate:** step by step.
- *Open printable board* gives a page to print at 100%.
- Enter the board size and press *Start*.
- Move the board around, covering the whole image, and press *Capture* (or Space), or turn on *Auto-capture*.
- Press *Calibrate & save*. You get an error figure and a plain-language quality rating.

**Settings:** team number, NetworkTables server (for simulation), video source (camera or a recording), resolution, field layout (*Welded* or *AndyMark*: check your event), YOLO model and upload, confidence, GPU/CPU, how often to re-check the pose, and whether other devices on the network may open the dashboard. Settings are saved to `settings.yaml` and applied immediately.

### Recording and replaying matches

1. Press **● Record** during a match.
2. Later, choose *Settings → Video source → Recording* and pick the file.

The whole pipeline runs on it exactly as it did live, so you can tune confidence, offsets and tracking at home without a field. *Pause* and *Restart* appear under the camera view.

## NetworkTables output

Table `/FieldMapper`, published every frame:

| topic | type | meaning |
|---|---|---|
| `x` | `double[]` | field X of each tracked piece, metres |
| `y` | `double[]` | field Y of each tracked piece, metres (same order as `x`) |
| `timestamp` | `double` | frame capture time in seconds **on the NT server clock**, so on the robot it is directly comparable with `Timer.getFPGATimestamp()` |
| `cameraPose` | `double[6]` | camera `x, y, z` (m), `roll, pitch, yaw` (rad) |
| `poseValid` | `boolean` | `x`/`y` are only meaningful when true |

Coordinates use the WPILib field frame with the **blue-alliance origin**: +X from the blue wall towards red, +Y to the left as seen from the blue driver station, +Z up. Set *Coordinate origin* to red if you want red-origin coordinates. `cameraPose` angles follow `Rotation3d(roll, pitch, yaw)` for the camera body (X out of the lens, Y left, Z up), so positive pitch means tilted down.

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
- The corners of every visible 36h11 tag go into **one** solvePnP against their 3D positions from the WPILib layout (SQPnP, then Levenberg–Marquardt refinement).
- A single tag uses IPPE: solutions that put the camera under the floor are dropped, then the one nearest the held pose wins. Ambiguous first solves are refused.
- Re-checks run every 6 s. With no pose yet, or while confirming a possible camera move, it retries every 0.5 s.
- A new solve is rejected if its reprojection error is high or it jumps too far from the held pose. Accepted solves are blended in (slerp), not snapped.
- If the camera really was bumped, three consecutive solves that agree with *each other* re-anchor the pose.

**Projection** (`field_mapper/geometry.py`)
1. Take the bottom-centre pixel of each detection box.
2. Undistort it.
3. Cast a ray from the camera pose and intersect it with the floor (z = 0).
4. Drop points that fall outside the field.

**Tracking** (`field_mapper/tracker.py`): greedy nearest-neighbour in field metres, per class.
- A piece must be seen 3 times before it is published.
- It stays published for up to 10 missed frames, so it doesn't flicker.

**App** (`field_mapper/service.py`, `field_mapper/web/`)
- The pipeline runs on a background thread.
- The FastAPI server streams the annotated camera image (MJPEG) and pushes state to the page 10× per second (server-sent events).
- No internet or CDN is needed.
- The app starts in any state (no camera, calibration or model yet) and picks things up as soon as they're fixed.

## Accuracy: what to expect

The camera views the floor at a very shallow angle, and that dominates everything. For camera height *h*, ground distance *d* and focal length *f* (pixels):

- **range error ≈ (h² + d²) / (h·f)** metres per pixel of error at the box's bottom edge
- **sideways error ≈ d / f** metres per pixel

| camera height | resolution (70° lens) | range error per px at 5 / 10 / 15 m |
|---|---|---|
| 2.0 m | 1280×720 | 1.6 / 5.7 / 12.5 cm |
| 2.0 m | 1920×1080 | 1.0 / 3.8 / 8.4 cm |
| 1.5 m | 1920×1080 | 1.3 / 5.0 / 11.0 cm |

- **Mount the camera as high as you're allowed, and use 1080p or more.**
- **Camera pitch matters most.** 0.1° of pitch error is about 9 cm at 10 m. Two or more tags per solve are far better than one, which is why *Tags needed per solve* defaults to 2.
- **Tags need enough pixels.** A 6.5 in tag is f·0.165/d px wide: at 720p that's ~30 px at 5 m and ~9 px across the full field.
- **Pieces off the floor project to the wrong place.** That includes pieces held by a robot, on a ledge or stacked: they land further away along the line of sight. Hidden pieces aren't seen.
- **The box bottom is the piece's near edge.** Set *Ground offset* to about the piece radius to report centres instead.

## Troubleshooting

| problem | fix |
|---|---|
| Browser didn't open | Go to http://localhost:5805 |
| Chip says **Model · CPU** | GPU PyTorch isn't working. Update the NVIDIA driver, then run `start.bat --reinstall` |
| **Camera problem** | *Settings → Find cameras*. Close other apps using the camera, or try another USB port |
| **Robot not connected** | Check the team number. The Driver Station must be running on this laptop, or set the NetworkTables server. For simulation use `127.0.0.1` |
| **Recalibrate** chip | The resolution changed to a different shape since calibrating. Calibrate again |
| Positions are off | Check that the yellow tag outlines line up in the camera view, the layout (*Welded* vs *AndyMark*) is right, and the ground offset is set |
| Port 5805 in use | Change `web.port` in `config.yaml` |

## Advanced

- `config.yaml` holds every setting with comments. The dashboard saves its changes to `settings.yaml`, which overrides it. Delete `settings.yaml` to go back to the defaults.
- Command line (inside the `.venv`): `python main.py`, with these options:
  - `--headless`: no dashboard, just run and publish;
  - `--replay video.mp4 [--loop] [--speed 0]`;
  - `--no-nt`;
  - `--port`, `--host`;
  - `--no-browser`.
- `python calibrate.py` calibrates from a terminal, either live or with `--images "dir/*.png"` / `--video file`.
- `start.bat --reinstall` redoes the dependency setup.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

- **Floor projection maths** (`tests/test_projection.py`):
  - round trips with and without lens distortion;
  - analytic cases;
  - horizon and field-bounds handling;
  - the pose conventions.
- **Localisation** against the real 2026 layout, including real AprilTag detection on rendered images.
- **Tracker.**
- **Browser calibration** on rendered, distorted checkerboards.
- **Config overrides.**
- **The web app end to end** on a generated replay.

## Layout

```
start.bat / start.sh    one-click start (first run installs everything)
launch.py               dependency setup + start
config.yaml             all settings (the dashboard writes overrides to settings.yaml)
main.py                 command-line entry point
calibrate.py            command-line calibration
field_mapper/
  app.py                CLI: dashboard or headless
  service.py            runs the pipeline thread; state + commands for the dashboard
  web/server.py         FastAPI: dashboard, camera stream, JSON API
  web/static/           the dashboard page (HTML/CSS/JS, no build step)
  pipeline.py           one frame: localise -> detect -> project -> track -> publish
  localizer.py          AprilTag detection, PnP, pose filter
  geometry.py           pose maths, undistortion, ray/floor intersection
  detector.py           Ultralytics YOLO wrapper
  tracker.py            nearest-neighbour tracker
  publisher.py          NetworkTables output
  calib_core.py         checkerboard calibration (dashboard + calibrate.py)
  capture.py            camera, replay, recording, camera search
  overlay.py            camera-image overlay
  config.py / calibration.py / field.py
tests/
```
