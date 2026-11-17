"""Command line entry point. Default: start the web dashboard and open it in the browser."""

from __future__ import annotations

import argparse
import logging
import socket
import time
import webbrowser
from pathlib import Path

from .config import load_config

log = logging.getLogger("field_mapper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FRC driver-station game piece mapper")
    p.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    p.add_argument("--replay", metavar="VIDEO", help="run on a video file instead of the live camera")
    p.add_argument("--speed", type=float, default=None, help="replay speed (1 = real time, 0 = as fast as possible)")
    p.add_argument("--loop", action="store_true", default=None, help="loop the replay video")
    p.add_argument("--calib", help="override camera.calibration_file")
    p.add_argument("--model", help="override detector.model_path")
    p.add_argument("--no-nt", action="store_true", help="don't publish to NetworkTables")
    p.add_argument("--headless", action="store_true", help="no web dashboard (just run and publish)")
    p.add_argument("--host", help="web dashboard address (default from config: 127.0.0.1)")
    p.add_argument("--port", type=int, help="web dashboard port (default from config: 5805)")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _already_running(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:  # OpenCV prints low-level warnings for every missing camera; we report problems ourselves
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    config_path = Path(args.config)
    if not config_path.exists():
        log.warning("%s not found - using defaults", config_path)
    cfg = load_config(config_path)

    from .service import MapperService

    svc = MapperService(
        config_path,
        replay=args.replay,
        replay_speed=args.speed,
        loop=args.loop,
        nt_enabled=not args.no_nt,
        calib_override=args.calib,
        model_override=args.model,
    )

    if args.headless:
        svc.start()
        try:
            while not svc.finished:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            svc.stop()
        return 0

    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    url = f"http://localhost:{port}/"
    if _already_running(port):
        log.info("FieldMapper is already running - opening %s", url)
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    import uvicorn

    from .web.server import create_app

    open_url = None if (args.no_browser or not cfg.web.open_browser) else url
    app = create_app(svc, open_browser_url=open_url)
    print(f"\n  FRC Field Mapper is running:  {url}\n  (close this window or press Ctrl+C to stop)\n", flush=True)
    if host not in ("127.0.0.1", "localhost"):
        log.info("dashboard reachable from other devices on port %d", port)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False, timeout_graceful_shutdown=2)
    return 0
