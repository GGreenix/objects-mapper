"""Greedy nearest-neighbour tracker in field coordinates.

Stops pieces flickering in and out: a track has to be seen ``min_hits`` times
before it is reported, and it keeps being reported (at its last position) for
up to ``max_missed_frames`` frames without a matching detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    x: float
    y: float
    class_id: int = 0
    confidence: float = 1.0


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    class_id: int
    confidence: float
    hits: int = 1
    misses: int = 0

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits


class NearestNeighbourTracker:
    def __init__(
        self,
        max_match_distance_m: float = 0.5,
        max_missed_frames: int = 10,
        min_hits: int = 3,
        smoothing_alpha: float = 0.5,
    ) -> None:
        self.max_dist = float(max_match_distance_m)
        self.max_missed = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self.alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, measurements: list[Measurement]) -> list[Track]:
        """Feed one frame of measurements; returns the confirmed (published) tracks."""
        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(self.tracks):
            for mi, m in enumerate(measurements):
                if m.class_id != t.class_id:
                    continue
                d = float(np.hypot(m.x - t.x, m.y - t.y))
                if d <= self.max_dist:
                    pairs.append((d, ti, mi))
        pairs.sort()

        used_t: set[int] = set()
        used_m: set[int] = set()
        for _, ti, mi in pairs:
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            t, m = self.tracks[ti], measurements[mi]
            t.x += self.alpha * (m.x - t.x)
            t.y += self.alpha * (m.y - t.y)
            t.confidence = m.confidence
            t.hits += 1
            t.misses = 0

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missed]

        for mi, m in enumerate(measurements):
            if mi not in used_m:
                self.tracks.append(Track(self._next_id, m.x, m.y, m.class_id, m.confidence))
                self._next_id += 1

        return self.confirmed()

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed(self.min_hits)]
