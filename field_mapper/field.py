"""AprilTag field layout: tag poses and their 3D corner positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import quaternion_to_rotation


@dataclass(frozen=True)
class TagPose:
    rotation: np.ndarray  # 3x3, tag frame -> field frame (tag +X out of the face, +Z up)
    position: np.ndarray  # (3,) tag centre


def tag_corners_local(tag_size: float) -> np.ndarray:
    """Tag corners in the WPILib tag frame, in AprilTag detector order.

    The detector returns corners counter-clockwise starting at the bottom-left
    as seen by a viewer facing the tag: bottom-left, bottom-right, top-right,
    top-left. Facing the tag you look along -X, so +Y is on your right.
    """
    h = tag_size / 2.0
    return np.array(
        [
            [0.0, -h, -h],  # bottom-left
            [0.0, h, -h],  # bottom-right
            [0.0, h, h],  # top-right
            [0.0, -h, h],  # top-left
        ]
    )


class FieldLayout:
    def __init__(
        self,
        tags: dict[int, TagPose],
        length: float,
        width: float,
        tag_size: float,
        ignore_ids: tuple[int, ...] | list[int] = (),
    ) -> None:
        ignore = set(int(i) for i in ignore_ids)
        self.tags = {int(k): v for k, v in tags.items() if int(k) not in ignore}
        self.length = float(length)
        self.width = float(width)
        self.tag_size = float(tag_size)
        local = tag_corners_local(self.tag_size)
        self._corners = {tid: local @ t.rotation.T + t.position for tid, t in self.tags.items()}

    @classmethod
    def load(
        cls, layout: str, tag_size: float, origin: str = "blue", ignore_ids: tuple[int, ...] | list[int] = ()
    ) -> "FieldLayout":
        """Load a built-in WPILib layout by name (e.g. ``k2026RebuiltWelded``) or a JSON file path."""
        import robotpy_apriltag as ra

        if Path(layout).suffix.lower() == ".json" or Path(layout).exists():
            if not Path(layout).exists():
                raise FileNotFoundError(f"field layout JSON not found: {layout}")
            wl = ra.AprilTagFieldLayout(str(layout))
        else:
            name = layout if layout.startswith("k") else f"k{layout}"
            if not hasattr(ra.AprilTagField, name):
                options = [n for n in ra.AprilTagField.__members__ if n.startswith("k20")]
                raise ValueError(f"unknown built-in field layout '{layout}'. Options: {', '.join(options)}")
            wl = ra.AprilTagFieldLayout.loadField(getattr(ra.AprilTagField, name))

        origin = origin.lower()
        if origin == "red":
            wl.setOrigin(ra.AprilTagFieldLayout.OriginPosition.kRedAllianceWallRightSide)
        elif origin == "blue":
            wl.setOrigin(ra.AprilTagFieldLayout.OriginPosition.kBlueAllianceWallRightSide)
        else:
            raise ValueError("field.origin must be 'blue' or 'red'")

        tags: dict[int, TagPose] = {}
        for tag in wl.getTags():
            pose = wl.getTagPose(tag.ID)  # respects the chosen origin
            q = pose.rotation().getQuaternion()
            tags[tag.ID] = TagPose(
                rotation=quaternion_to_rotation(q.W(), q.X(), q.Y(), q.Z()),
                position=np.array([pose.X(), pose.Y(), pose.Z()]),
            )
        return cls(tags, wl.getFieldLength(), wl.getFieldWidth(), tag_size, ignore_ids)

    def __contains__(self, tag_id: int) -> bool:
        return tag_id in self.tags

    def corners(self, tag_id: int) -> np.ndarray:
        """(4, 3) field-frame corners of a tag in detector order."""
        return self._corners[tag_id]

    def in_bounds(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        from .geometry import in_field_bounds

        return in_field_bounds(xy, self.length, self.width, margin)
