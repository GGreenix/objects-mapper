"""The web app end to end: service + HTTP API on a generated replay video.

No model is configured, so this also checks the app keeps working (localising,
serving the dashboard) with parts of the setup missing.
"""

import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from field_mapper.calibration import CameraIntrinsics, save_intrinsics  # noqa: E402
from field_mapper.capture import Recorder, VideoSource  # noqa: E402
from field_mapper.field import FieldLayout  # noqa: E402
from field_mapper.geometry import CameraPose  # noqa: E402
from field_mapper.service import MapperService  # noqa: E402
from field_mapper.web.server import create_app  # noqa: E402

from .synthetic import make_intrinsics, render_tags  # noqa: E402

TRUE_POSE = CameraPose.look_at([-0.6, 4.0, 1.9], [10.0, 4.0, 0.0])


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project folder with config.yaml, calib.yaml and a short replay of the field tags."""
    root = tmp_path_factory.mktemp("project")
    k = make_intrinsics()
    field = FieldLayout.load("k2026RebuiltWelded", 0.1651)
    img, _ = render_tags(field, TRUE_POSE, k)
    rec = Recorder(root / "recordings" / "replay.avi", 30, (img.shape[1], img.shape[0]))
    for _ in range(45):
        rec.write(img)
    rec.close()
    save_intrinsics(root / "calib.yaml", CameraIntrinsics(k, np.zeros(5), img.shape[1], img.shape[0], 0.1))
    (root / "config.yaml").write_text(
        "team_number: 0\n"
        "source: {type: video, video_path: recordings/replay.avi, loop: true, speed: 0}\n"
        "networktables: {enabled: false}\n"
        "detector: {model_path: models/none.pt}\n"
    )
    return root


def wait_for(client, predicate, timeout=20.0):
    deadline = time.time() + timeout
    st = None
    while time.time() < deadline:
        st = client.get("/api/state").json()
        if predicate(st):
            return st
        time.sleep(0.1)
    raise AssertionError(f"condition not met; last state: {st}")


def test_recorder_round_trip(tmp_path):
    rec = Recorder(tmp_path / "r.avi", 15, (64, 48))
    for i in range(10):
        rec.write(np.full((48, 64, 3), i * 20, np.uint8))
    rec.write(np.zeros((10, 10, 3), np.uint8))  # wrong size: ignored
    rec.close()
    assert rec.frames == 10
    src = VideoSource(tmp_path / "r.avi", speed=0)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    assert len(frames) == 10
    assert frames[-1].timestamp == pytest.approx(9 / 15)
    src.restart()
    assert src.read().timestamp == pytest.approx(10 / 15)  # time keeps going forwards after a restart


def test_web_app_end_to_end(project):
    svc = MapperService(project / "config.yaml", nt_enabled=False)
    with TestClient(create_app(svc)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Field Mapper" in page.text
        assert client.get("/static/app.js").status_code == 200

        st = wait_for(client, lambda s: s["pose"]["valid"])
        assert st["mapping"] and st["calibration"]["ok"]
        assert not st["model"]["ok"] and "no model file" in st["model"]["error"]
        assert abs(st["pose"]["x"] + 0.6) < 0.05 and abs(st["pose"]["z"] - 1.9) < 0.05
        assert len(st["used_tags"]) >= 4 and len(st["footprint"]) > 10
        assert st["camera"]["type"] == "video" and st["camera"]["status"] == "ok"

        field = client.get("/api/field").json()
        assert field["ok"] and len(field["tags"]) == 32

        snap = client.get("/snapshot.jpg")
        assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(snap.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (720, 1280)

        settings = client.get("/api/settings").json()
        assert "recordings/replay.avi" in settings["options"]["recordings"]
        assert "k2026RebuiltWelded" in settings["options"]["layouts"]

        bad = client.post("/api/settings", json={"team_number": "abc"})
        assert bad.status_code == 400 and "team_number" in bad.json()["error"]
        assert client.post("/api/settings", json={"team_number": 1234}).json()["ok"]
        assert (project / "settings.yaml").exists()
        wait_for(client, lambda s: s["nt"]["team"] == 1234)

        assert client.post("/api/action/pause", json={}).json()["ok"]
        assert client.get("/api/state").json()["camera"]["paused"]
        assert client.post("/api/action/resume", json={}).json()["ok"]
        assert client.post("/api/action/record_start", json={}).status_code == 400  # replay, not a camera
        assert client.post("/api/action/nope", json={}).status_code == 404

        up = client.put("/api/models/fuel.pt", content=b"not really a model")
        assert up.json()["path"] == "models/fuel.pt"
        assert client.put("/api/models/x.exe", content=b"x").status_code == 400

        assert client.post("/api/calibration/capture", json={}).status_code == 400  # not calibrating
        assert client.post("/api/calibration/start", json={"cols": 9, "rows": 6, "square_mm": 25}).json()["ok"]
        st = wait_for(client, lambda s: s["calibrating"]["active"] and s["calibrating"]["message"])
        assert not st["mapping"]
        assert client.post("/api/calibration/stop", json={}).json()["ok"]
        wait_for(client, lambda s: s["mapping"] and not s["calibrating"]["active"])

        assert client.get("/board.png?cols=9&rows=6&square_mm=25").status_code == 200
        assert client.get("/board.png?cols=99").status_code == 400


def test_uploading_over_the_default_model_path_loads_it(project, monkeypatch):
    """Regression: the default path is models/best.pt; uploading best.pt must load it, not be skipped."""
    import field_mapper.detector as det_mod

    class FakeDetector:
        def __init__(self, cfg, path):
            self.cfg, self.device, self.half, self.names = cfg, "cpu", False, {0: "fuel"}

        def set_thresholds(self, confidence, iou=None):
            pass

        def detect(self, img):
            return []

    monkeypatch.setattr(det_mod, "YoloDetector", FakeDetector)
    (project / "settings.yaml").unlink(missing_ok=True)
    svc = MapperService(project / "config.yaml", nt_enabled=False)
    svc.cfg.detector.model_path = "models/best.pt"
    with TestClient(create_app(svc)) as client:
        svc.apply_settings({"detector": {"model_path": "models/best.pt"}})
        wait_for(client, lambda s: (s["model"]["error"] or "").startswith("no model file"))
        r = client.put("/api/models/best.pt", content=b"weights").json()
        assert client.post("/api/settings", json={"detector": {"model_path": r["path"]}}).json()["ok"]
        st = wait_for(client, lambda s: s["model"]["ok"])
        assert st["model"]["error"] is None
