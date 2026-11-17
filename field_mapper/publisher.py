"""NetworkTables (NT4) output under /FieldMapper."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class NTPublisher:
    """Publishes x[], y[], timestamp, cameraPose[], poseValid.

    ``timestamp`` is the frame capture time in seconds on the NT server's clock
    (on a roboRIO that is the FPGA clock, i.e. comparable with
    ``Timer.getFPGATimestamp()``). Before the time sync with the server is
    established it falls back to this machine's NT clock.
    """

    def __init__(self, team: int, table: str = "FieldMapper", server: str = "", client_name: str = "FieldMapper") -> None:
        import ntcore

        self._ntcore = ntcore
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.startClient4(client_name)
        if server:
            self.inst.setServer(server)
            log.info("NetworkTables: connecting to %s", server)
        else:
            if team <= 0:
                log.warning("team_number is not set; NetworkTables will only find the robot via the Driver Station")
            else:
                self.inst.setServerTeam(team)
            self.inst.startDSClient()
            log.info("NetworkTables: team %d (+ Driver Station)", team)

        t = self.inst.getTable(table)
        opts = ntcore.PubSubOptions(periodic=0.01, sendAll=True, keepDuplicates=True)
        self._x = t.getDoubleArrayTopic("x").publish(opts)
        self._y = t.getDoubleArrayTopic("y").publish(opts)
        self._ts = t.getDoubleTopic("timestamp").publish(opts)
        self._pose = t.getDoubleArrayTopic("cameraPose").publish(opts)
        self._valid = t.getBooleanTopic("poseValid").publish(opts)
        self._valid.set(False)
        self._was_connected = False

    @property
    def connected(self) -> bool:
        return self.inst.isConnected()

    def server_time_of(self, capture_monotonic: float) -> float:
        """Convert a ``time.monotonic()`` capture instant to NT server seconds."""
        now_local_us = self._ntcore._now()
        age_us = (time.monotonic() - capture_monotonic) * 1e6
        offset = self.inst.getServerTimeOffset() or 0
        return (now_local_us - age_us + offset) / 1e6

    def publish(self, xs: list[float], ys: list[float], capture_monotonic: float, camera_pose: list[float], pose_valid: bool) -> float:
        connected = self.connected
        if connected != self._was_connected:
            log.info("NetworkTables %s", "connected" if connected else "disconnected")
            self._was_connected = connected
        ts = self.server_time_of(capture_monotonic)
        self._x.set(list(xs))
        self._y.set(list(ys))
        self._ts.set(ts)
        self._pose.set(list(camera_pose))
        self._valid.set(bool(pose_valid))
        self.inst.flush()
        return ts

    def close(self) -> None:
        try:
            self._valid.set(False)
            self.inst.flush()
            self.inst.stopDSClient()
            self.inst.stopClient()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


class NullPublisher:
    connected = False

    def publish(self, xs, ys, capture_monotonic, camera_pose, pose_valid) -> float:
        return capture_monotonic

    def close(self) -> None:
        pass
