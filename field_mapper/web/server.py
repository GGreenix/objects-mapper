"""FastAPI app: serves the dashboard, the camera stream and a small JSON API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..calib_core import printable_board
from ..service import MapperService

log = logging.getLogger("field_mapper.web")
STATIC = Path(__file__).parent / "static"


def _clean(o):
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _json(o) -> str:
    return json.dumps(_clean(o), separators=(",", ":"))


def create_app(svc: MapperService, open_browser_url: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start(defer_model_load_s=2.0)
        if open_browser_url:
            threading.Timer(1.0, webbrowser.open, [open_browser_url]).start()
        try:
            yield
        finally:
            app.state.stopping = True
            await asyncio.to_thread(svc.stop)

    app = FastAPI(title="FRC Field Mapper", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.stopping = False
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def fail(e: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(e))

    # ------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/stream.mjpg")
    async def stream(request: Request, field: int = 1, det: int = 1, width: int = 0):
        interval = 1.0 / max(1.0, float(svc.cfg.web.stream_fps))

        async def gen():
            last_index, last_sent = None, 0.0
            loop = asyncio.get_running_loop()
            while not app.state.stopping:
                if await request.is_disconnected():
                    break
                idx = svc.frame_index
                now = loop.time()
                if idx != last_index or now - last_sent > 0.5:
                    data = await asyncio.to_thread(svc.render_jpeg, width or None, None, bool(field), bool(det))
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    last_index, last_sent = idx, now
                await asyncio.sleep(interval)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/snapshot.jpg")
    async def snapshot(field: int = 1, det: int = 1):
        data = await asyncio.to_thread(svc.render_jpeg, 10000, 92, bool(field), bool(det))
        return Response(data, media_type="image/jpeg",
                         headers={"Content-Disposition": 'inline; filename="fieldmapper.jpg"'})

    # --------------------------------------------------------------- state
    @app.get("/api/state")
    async def state():
        st = await asyncio.to_thread(svc.state)
        return Response(_json(st), media_type="application/json")

    @app.get("/api/events")
    async def events(request: Request, log_after: int = 0):
        async def gen():
            seq = log_after
            while not app.state.stopping:
                if await request.is_disconnected():
                    break
                st = await asyncio.to_thread(svc.state)
                logs = svc.logs.since(seq)
                if logs:
                    seq = logs[-1]["seq"]
                st["logs"] = logs
                yield f"data: {_json(st)}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/field")
    async def field_view():
        return Response(_json(svc.field_view()), media_type="application/json")

    # ------------------------------------------------------------ settings
    @app.get("/api/settings")
    async def get_settings():
        return Response(_json(await asyncio.to_thread(svc.settings_view)), media_type="application/json")

    @app.post("/api/settings")
    async def post_settings(request: Request):
        values = await request.json()
        if not isinstance(values, dict):
            raise HTTPException(400, "expected a JSON object")
        try:
            svc.apply_settings(values)
        except ValueError as e:
            raise fail(e) from e
        return {"ok": True}

    @app.get("/api/cameras")
    async def cameras():
        found = await asyncio.to_thread(svc.probe_cameras)
        for c in found:
            jpeg = c.pop("jpeg", None)
            c["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode() if jpeg else None
        return found

    @app.put("/api/models/{filename}")
    async def upload_model(filename: str, request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            path = await asyncio.to_thread(svc.save_model, filename, data)
        except ValueError as e:
            raise fail(e) from e
        return {"ok": True, "path": path}

    # ------------------------------------------------------------- actions
    @app.post("/api/action/{name}")
    async def action(name: str, request: Request):
        body = {}
        if request.headers.get("content-length", "0") != "0":
            try:
                body = await request.json()
            except json.JSONDecodeError:
                body = {}
        try:
            result = None
            if name == "relocalize":
                svc.relocalize()
            elif name == "reset_tracker":
                svc.reset_tracker()
            elif name == "pause":
                svc.set_paused(True)
            elif name == "resume":
                svc.set_paused(False)
            elif name == "restart_replay":
                svc.restart_replay()
            elif name == "record_start":
                result = svc.start_recording()
            elif name == "record_stop":
                svc.stop_recording()
            elif name == "confidence":
                svc.set_confidence(float(body["value"]))
            else:
                raise HTTPException(404, f"unknown action {name}")
        except (ValueError, KeyError, TypeError) as e:
            raise fail(e) from e
        return {"ok": True, "result": result}

    # --------------------------------------------------------- calibration
    @app.post("/api/calibration/{name}")
    async def calibration(name: str, request: Request):
        body = {}
        if request.headers.get("content-length", "0") != "0":
            body = await request.json()
        session = svc.calib
        try:
            if name == "start":
                c = svc.cfg.calibration
                svc.start_calibration(int(body.get("cols", c.board_cols)), int(body.get("rows", c.board_rows)),
                                      float(body.get("square_mm", c.square_size_m * 1000)),
                                      int(body.get("min_views", c.min_frames)))
            elif name == "stop":
                svc.stop_calibration()
            elif session is None:
                raise ValueError("calibration is not running")
            elif name == "capture":
                session.request_capture()
            elif name == "auto":
                session.set_auto(bool(body.get("on", True)))
            elif name == "undo":
                session.undo()
            elif name == "clear":
                session.clear()
            elif name == "solve":
                if not session.start_solve():
                    raise ValueError(f"need at least {session.min_views} views")
            else:
                raise HTTPException(404, f"unknown calibration command {name}")
        except ValueError as e:
            raise fail(e) from e
        return {"ok": True}

    @app.get("/board.png")
    async def board_png(cols: int = 9, rows: int = 6, square_mm: float = 25.0):
        if not (2 <= cols <= 30 and 2 <= rows <= 30 and 5 <= square_mm <= 200):
            raise HTTPException(400, "board size out of range")
        img = printable_board(cols, rows, square_mm)
        return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")

    @app.get("/board", response_class=HTMLResponse)
    async def board_page(cols: int = 9, rows: int = 6, square_mm: float = 25.0):
        sq_px = round(square_mm / 25.4 * 300)
        w_mm = ((cols + 1) * sq_px + 2 * sq_px) / 300 * 25.4
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Calibration board</title>
<style>@page {{ margin: 8mm; }} body {{ margin: 0; font-family: system-ui, sans-serif; }}
img {{ width: {w_mm:.2f}mm; display: block; }} .hint {{ padding: 12px; }} @media print {{ .hint {{ display: none; }} }}</style>
</head><body><div class="hint">Print this page at <b>100% scale</b> (turn off "fit to page"), glue it to something flat,
then measure one square and enter that size in the Calibrate tab. <button onclick="print()">Print</button></div>
<img src="/board.png?cols={cols}&rows={rows}&square_mm={square_mm}" alt="checkerboard"></body></html>"""

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)

    return app
