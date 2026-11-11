"""
Data types representing OptiTrack/NatNet objects.

These are the internal data structures used throughout the tool.
They are populated by the NatNet client and consumed by the geometry
and scene modules. All positions are in OptiTrack's native Y-up coordinate
frame at this stage — conversion to Z-up happens in geometry/transform.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

@dataclass
class Vector3:
    """A 3D position or offset in meters."""
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class Quaternion:
    """
    Orientation as a unit quaternion.
    Scalar-last convention: [qx, qy, qz, qw] — matches ROS and scipy.
    """
    qx: float
    qy: float
    qz: float
    qw: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)

    def as_list(self) -> list[float]:
        return [self.qx, self.qy, self.qz, self.qw]

    def __repr__(self) -> str:
        return f"Quaternion({self.qx:.4f}, {self.qy:.4f}, {self.qz:.4f}, {self.qw:.4f})"


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------

@dataclass
class Marker:
    """
    A single reflective marker as reported by NatNet.

    Position is in world space (the OptiTrack capture volume origin),
    Y-up frame, metres.

    marker_id:   NatNet's unique ID for this marker within its rigid body.
    position:    World-space position in the OptiTrack Y-up frame.
    size:        Reported marker diameter in metres (optional, not always
                 populated by NatNet depending on streaming settings).
    residual:    Reconstruction residual in metres — how well the marker
                 position was solved from camera views. Lower is better.
                 Typical good values are under 0.001 m (1 mm).
    """
    marker_id: int
    position: Vector3
    size: float = 0.0
    residual: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Marker(id={self.marker_id}, pos={self.position}, "
            f"residual={self.residual:.4f}m)"
        )


# ---------------------------------------------------------------------------
# Rigid body
# ---------------------------------------------------------------------------

@dataclass
class RigidBody:
    """
    A rigid body as defined in Motive and streamed via NatNet.

    body_id:    NatNet's integer ID assigned in Motive.
    name:       Human-readable name from Motive (e.g. 'wall_left').
                This becomes the STL filename and the key in scene.json.
    position:   Centroid position in world space, OptiTrack Y-up frame.
    orientation: Body orientation as a quaternion, scalar-last.
    markers:    The reflective markers belonging to this body, in world
                space. This is what we use to reconstruct geometry.
    tracking_valid: False if Motive lost tracking this frame (e.g. markers
                    occluded). We skip geometry reconstruction for invalid
                    frames.
    mean_error: Mean marker reconstruction error in metres. A quality
                indicator — values above ~0.005 m suggest poor tracking.
    """
    body_id: int
    name: str
    position: Vector3
    orientation: Quaternion
    markers: list[Marker] = field(default_factory=list)
    tracking_valid: bool = True
    mean_error: float = 0.0

    @property
    def marker_count(self) -> int:
        return len(self.markers)

    @property
    def marker_positions(self) -> list[tuple[float, float, float]]:
        """Convenience: all marker positions as a list of (x, y, z) tuples."""
        return [m.position.as_tuple() for m in self.markers]

    def __repr__(self) -> str:
        return (
            f"RigidBody(id={self.body_id}, name='{self.name}', "
            f"markers={self.marker_count}, valid={self.tracking_valid})"
        )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class CameraInfo:
    """
    An OptiTrack camera as reported by NatNet.

    camera_id:  NatNet's integer ID.
    name:       Camera name from Motive (e.g. 'Prime 13W 1').
    position:   Camera position in world space, OptiTrack Y-up frame.
                For boundary detection we use the XZ floor projection
                (X and Z in OptiTrack = X and Y in Z-up world frame).
    is_corner:  Set to True after auto-detection identifies this camera
                as one of the 4 extreme corners of the camera array.
                Used to define the room boundary rectangle.
    """
    camera_id: int
    name: str
    position: Vector3
    is_corner: bool = False

    def __repr__(self) -> str:
        corner_tag = " [corner]" if self.is_corner else ""
        return f"CameraInfo(id={self.camera_id}, name='{self.name}', pos={self.position}{corner_tag})"



# ---------------------------------------------------------------------------
# Frame — one complete snapshot from NatNet
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """
    A single data frame received from NatNet.

    NatNet streams frames continuously (typically at 100–240 Hz).
    For obstacle building we only need a stable snapshot, so the client
    will capture one frame on demand rather than processing every frame.

    frame_number:  NatNet's monotonically increasing frame counter.
    timestamp:     Time in seconds since streaming started.
    rigid_bodies:  All rigid bodies tracked in this frame.
    cameras:       Camera info — only populated on the first frame or on
                   explicit request (NatNet sends this less frequently).
    is_recording:  Whether Motive is currently recording (informational).
    """
    frame_number: int
    timestamp: float
    rigid_bodies: list[RigidBody] = field(default_factory=list)
    cameras: list[CameraInfo] = field(default_factory=list)
    is_recording: bool = False

    property
    def rigid_body_count(self) -> int:
        return len(self.rigid_bodies)

    def get_body_by_name(self, name: str) -> Optional[RigidBody]:
        """Return the first rigid body with the given name, or None."""
        for body in self.rigid_bodies:
            if body.name == name:
                return body
        return None

    def get_body_by_id(self, body_id: int) -> Optional[RigidBody]:
        """Return the rigid body with the given ID, or None."""
        for body in self.rigid_bodies:
            if body.body_id == body_id:
                return body
        return None

    def __repr__(self) -> str:
        return (
            f"Frame(#{self.frame_number}, t={self.timestamp:.3f}s, "
            f"bodies={self.rigid_body_count})"
        )