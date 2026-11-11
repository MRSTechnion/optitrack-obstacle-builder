"""
Coordinate frame conversion between OptiTrack and world frames.

OptiTrack native frame (what NatNet reports):
    X → right
    Y → up
    Z → toward viewer (out of the volume)
    Units: metres

World frame (what this tool uses, and what scene.json stores):
    X → right  (unchanged)
    Y → into the volume  (was OptiTrack Z, sign flipped)
    Z → up     (was OptiTrack Y)
    Units: metres

The conversion is:
    world_x =  optitrack_x
    world_y = -optitrack_z
    world_z =  optitrack_y

For orientations (quaternions) the same axis relabelling applies,
implemented as a pre-multiplication by a fixed rotation quaternion.

Y-up export mode simply skips the conversion and passes coordinates
through unchanged — it is the caller's responsibility to pass the
correct mode flag.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..natnet.types import CameraInfo, Frame, Marker, Quaternion, RigidBody, Vector3


# ---------------------------------------------------------------------------
# Axis remapping constants
# ---------------------------------------------------------------------------

# The rotation that takes a vector from OptiTrack frame to world (Z-up) frame.
# Rotating +90 degrees around the X axis swaps Y and Z:
#   new_Y = old_Z  (with sign flip to maintain right-handedness)
#   new_Z = old_Y
#
# As a rotation matrix this is:
#   [1,  0,  0]
#   [0,  0,  1]
#   [0, -1,  0]
#
# We store it as a scipy Rotation so we can compose it with body orientations.
_R_OPTITRACK_TO_WORLD = Rotation.from_matrix(np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=float))

_R_OPTITRACK_TO_WORLD_QUAT = _R_OPTITRACK_TO_WORLD.as_quat()  # [qx,qy,qz,qw]


# ---------------------------------------------------------------------------
# Single-vector conversion
# ---------------------------------------------------------------------------

def convert_position(v: Vector3, z_up: bool = True) -> Vector3:
    """
    Convert a position from OptiTrack frame to world frame.

    Parameters
    ----------
    v     : Vector3 in OptiTrack native coordinates (Y-up)
    z_up  : True  → convert to Z-up world frame (default)
            False → pass through unchanged (Y-up export mode)

    Returns
    -------
    Vector3 in world frame
    """
    if not z_up:
        return Vector3(v.x, v.y, v.z)

    # world_x =  optitrack_x
    # world_y = -optitrack_z
    # world_z =  optitrack_y
    return Vector3(
        x =  v.x,
        y = -v.z,
        z =  v.y,
    )


def convert_quaternion(q: Quaternion, z_up: bool = True) -> Quaternion:
    """
    Convert an orientation quaternion from OptiTrack frame to world frame.

    The body's orientation must be re-expressed in the new axis convention.
    We do this by pre-multiplying with the frame-change rotation:

        q_world = R_frame_change * q_optitrack

    Parameters
    ----------
    q     : Quaternion in OptiTrack native frame, scalar-last [qx,qy,qz,qw]
    z_up  : True  → convert to Z-up (default)
            False → pass through unchanged

    Returns
    -------
    Quaternion in world frame, scalar-last [qx,qy,qz,qw]
    """
    if not z_up:
        return Quaternion(q.qx, q.qy, q.qz, q.qw)

    q_body  = Rotation.from_quat([q.qx, q.qy, q.qz, q.qw])
    q_world = _R_OPTITRACK_TO_WORLD * q_body
    qx, qy, qz, qw = q_world.as_quat()   # scipy scalar-last convention

    return Quaternion(qx, qy, qz, qw)


# ---------------------------------------------------------------------------
# Marker positions: world-space → body-local frame
# ---------------------------------------------------------------------------

def markers_to_local_frame(
    marker_positions: list[tuple[float, float, float]],
    body_position:    tuple[float, float, float],
    body_quaternion:  tuple[float, float, float, float],
) -> np.ndarray:
    """
    Transform world-space marker positions into the rigid body's local frame.

    This is the step that makes the geometry independent of where the obstacle
    sits in the room. The resulting point cloud is centred at (0,0,0) and
    oriented along the body's own axes — ready for convex hull computation
    and STL export.

    All inputs should already be in world (Z-up) frame — call convert_position
    and convert_quaternion on them first.

    Parameters
    ----------
    marker_positions : list of (x, y, z) in world frame
    body_position    : (x, y, z) centroid in world frame
    body_quaternion  : (qx, qy, qz, qw) body orientation in world frame

    Returns
    -------
    np.ndarray of shape (N, 3) — marker positions in body-local frame
    """
    pts    = np.array(marker_positions, dtype=float)   # (N, 3)
    centre = np.array(body_position,    dtype=float)   # (3,)

    # Translate so centroid is at origin
    pts_centred = pts - centre

    # Rotate into body frame by applying the inverse of the body orientation
    R_body = Rotation.from_quat(list(body_quaternion))
    pts_local = R_body.inv().apply(pts_centred)

    return pts_local   # shape (N, 3)


# ---------------------------------------------------------------------------
# Full frame conversion
# ---------------------------------------------------------------------------

def convert_frame(frame: Frame, z_up: bool = True) -> Frame:
    """
    Return a new Frame with all positions and orientations converted to
    the world frame. The original Frame is not modified.

    This is the main entry point for the geometry pipeline:

        raw_frame     = client.capture_frame()
        world_frame   = convert_frame(raw_frame, z_up=True)
        # now pass world_frame to hull.py

    Parameters
    ----------
    frame : Frame in OptiTrack native Y-up coordinates
    z_up  : True (default) → convert to Z-up world frame
            False           → keep Y-up (for Y-up export mode)
    """
    converted_bodies = []
    for body in frame.rigid_bodies:
        converted_bodies.append(_convert_rigid_body(body, z_up))

    converted_cameras = []
    for cam in frame.cameras:
        converted_cameras.append(_convert_camera(cam, z_up))

    # Return a new Frame — dataclasses are mutable but we avoid mutating
    # the original so the caller can keep both for debugging if needed.
    from ..natnet.types import Frame as _Frame
    return _Frame(
        frame_number = frame.frame_number,
        timestamp    = frame.timestamp,
        rigid_bodies = converted_bodies,
        cameras      = converted_cameras,
        is_recording = frame.is_recording,
    )


def _convert_rigid_body(body: RigidBody, z_up: bool) -> RigidBody:
    """Convert one rigid body and all its markers to world frame."""
    world_pos = convert_position(body.position, z_up)
    world_ori = convert_quaternion(body.orientation, z_up)

    world_markers = []
    for m in body.markers:
        world_markers.append(Marker(
            marker_id = m.marker_id,
            position  = convert_position(m.position, z_up),
            size      = m.size,
            residual  = m.residual,
        ))

    return RigidBody(
        body_id        = body.body_id,
        name           = body.name,
        position       = world_pos,
        orientation    = world_ori,
        markers        = world_markers,
        tracking_valid = body.tracking_valid,
        mean_error     = body.mean_error,
    )


def _convert_camera(cam: CameraInfo, z_up: bool) -> CameraInfo:
    """Convert one camera position to world frame."""
    return CameraInfo(
        camera_id = cam.camera_id,
        name      = cam.name,
        position  = convert_position(cam.position, z_up),
        is_corner = cam.is_corner,
    )