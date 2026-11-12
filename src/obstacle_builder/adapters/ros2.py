"""
ROS 2 adapter — publish an exported session to ROS 2.

    obstacle-builder-ros2 SESSION [--live] [--frame-id map]

Publishes, in the frame given by --frame-id (default "map", which must
match the OptiTrack origin in your TF tree):

  ~/markers  visualization_msgs/MarkerArray — the obstacle hulls, their
             names, the boundary walls and the floor outline, for RViz
  /planning_scene  moveit_msgs/PlanningScene (diff) — each obstacle and
             wall as a MoveIt collision object (mesh). Only if moveit_msgs is
             installed; resent whenever a new subscriber (move_group) appears
  ~/octomap  octomap_msgs/Octomap — the session's octomap.bt, if present

Markers and the OctoMap are latched (transient local): subscribers that start
later still receive them.

With --live the node also connects to Motive (NatNet) and moves each obstacle
to the live pose of the rigid body with the same name, so obstacles moved
after export are followed. Shapes stay as exported. The OctoMap is not
updated live: it shows the saved poses.

This module has no ROS imports, so it is usable (and tested) without ROS;
the node itself is in ros2_node.py and needs a sourced ROS 2 environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..natnet.types import Frame
from ..scene.scene_writer import LoadedSession, load_session
from .gazebo import _PALETTE

logger = logging.getLogger(__name__)

DEFAULT_FRAME_ID   = "map"
DEFAULT_LIVE_RATE  = 10.0     # Hz
POSITION_TOL_M     = 0.001    # live changes smaller than this are not republished
ROTATION_TOL_DEG   = 0.1


@dataclass
class SceneObject:
    """One solid to publish: a triangle mesh in its own frame, placed at a pose."""
    name:        str
    vertices:    np.ndarray                      # (N, 3), object frame
    faces:       np.ndarray                      # (M, 3) vertex indices
    position:    tuple[float, float, float]
    orientation: tuple[float, float, float, float]   # qx, qy, qz, qw
    color:       tuple[float, float, float, float]   # RGBA 0-1
    is_wall:     bool = False


@dataclass
class SceneContent:
    obstacles:     list[SceneObject]
    walls:         list[SceneObject]
    floor_polygon: np.ndarray | None            # (K, 3), counter-clockwise
    octomap_path:  Path | None


_WALL_RGBA = (0.70, 0.70, 0.75, 0.5)


def scene_content(session: LoadedSession | Path | str, *, walls: bool = True,
                  octomap: bool = True) -> SceneContent:
    """Everything the node publishes, read from a session."""
    s = session if isinstance(session, LoadedSession) else load_session(Path(session))
    obstacles = [
        SceneObject(
            name        = body.name,
            vertices    = np.asarray(s.hulls[body.name].vertices, dtype=float),
            faces       = np.asarray(s.hulls[body.name].faces, dtype=int),
            position    = body.position.as_tuple(),
            orientation = body.orientation.as_tuple(),
            color       = (*_PALETTE[i % len(_PALETTE)], 0.9),
        )
        for i, body in enumerate(s.bodies)
    ]
    wall_objs, floor = [], None
    if s.boundary is not None:
        floor = np.asarray(s.boundary.floor_vertices, dtype=float)
        if walls:
            wall_objs = [
                SceneObject(f"boundary_wall_{i}", np.asarray(v, dtype=float), np.asarray(f, dtype=int),
                            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), _WALL_RGBA, is_wall=True)
                for i, (v, f) in enumerate(s.boundary.wall_meshes)
            ]
    bt = s.session_dir / "octomap.bt"
    return SceneContent(obstacles, wall_objs, floor, bt if octomap and bt.exists() else None)


def read_octomap_file(path: Path | str) -> tuple[str, float, bytes]:
    """
    Split a binary OctoMap file (.bt) into (tree id, resolution, data) —
    the fields of an octomap_msgs/Octomap message (binary=True). The data
    is the file's body after its text header.
    """
    raw = Path(path).read_bytes()
    marker = b"\ndata\n"
    end = raw.find(marker)
    if not raw.startswith(b"# Octomap OcTree binary file") or end < 0:
        raise ValueError(f"Not a binary OctoMap file: {path}")
    tree_id, res = "OcTree", None
    for line in raw[:end].decode("ascii", "replace").splitlines():
        key, _, value = line.partition(" ")
        if key == "id":
            tree_id = value.strip()
        elif key == "res":
            res = float(value)
    if res is None:
        raise ValueError(f"OctoMap file has no resolution: {path}")
    return tree_id, res, raw[end + len(marker):]


class LivePoseTracker:
    """
    Follows obstacles in live frames: matches rigid bodies to obstacles by
    name and reports the ones that moved by more than the tolerances since
    they were last reported. Untracked bodies keep their last pose.
    """

    def __init__(self, poses: dict[str, tuple[tuple, tuple]],
                 position_tol: float = POSITION_TOL_M, rotation_tol_deg: float = ROTATION_TOL_DEG):
        self.poses = dict(poses)              # name → (position, orientation)
        self._pos_tol = position_tol
        self._rot_tol = np.radians(rotation_tol_deg)

    def update(self, world_frame: Frame) -> dict[str, tuple[tuple, tuple]]:
        """`world_frame` in the session frame (Z-up). Returns the moved obstacles."""
        moved = {}
        for body in world_frame.rigid_bodies:
            if body.name not in self.poses or not body.tracking_valid:
                continue
            pos, quat = body.position.as_tuple(), body.orientation.as_tuple()
            old_pos, old_quat = self.poses[body.name]
            dp = np.linalg.norm(np.subtract(pos, old_pos))
            dr = (Rotation.from_quat(old_quat).inv() * Rotation.from_quat(quat)).magnitude()
            if dp > self._pos_tol or dr > self._rot_tol:
                self.poses[body.name] = (pos, quat)
                moved[body.name] = (pos, quat)
        return moved


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from rclpy.utilities import remove_ros_args
    except ImportError:
        print("error: ROS 2 (rclpy) not found. Source your ROS 2 environment first, e.g.\n"
              "  source /opt/ros/humble/setup.bash", file=sys.stderr)
        return 1
    from ..natnet.client import DEFAULT_SERVER_IP

    parser = argparse.ArgumentParser(
        prog="obstacle-builder-ros2",
        description="Publish an exported session to ROS 2 (RViz markers, MoveIt, OctoMap).",
    )
    parser.add_argument("session", type=Path, help="session directory, or its scene.json")
    parser.add_argument("--frame-id", default=DEFAULT_FRAME_ID,
                        help=f"frame of the published data (default {DEFAULT_FRAME_ID})")
    parser.add_argument("--live", action="store_true",
                        help="follow the obstacles' live poses from Motive")
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP,
                        help=f"Motive PC address for --live (default {DEFAULT_SERVER_IP})")
    parser.add_argument("--rate", type=float, default=DEFAULT_LIVE_RATE,
                        help=f"live update rate in Hz (default {DEFAULT_LIVE_RATE:g})")
    parser.add_argument("--no-walls", action="store_true", help="do not publish boundary walls")
    parser.add_argument("--no-moveit", action="store_true", help="do not publish to MoveIt")
    parser.add_argument("--no-octomap", action="store_true", help="do not publish octomap.bt")
    args = parser.parse_args(remove_ros_args(["prog", *argv])[1:])

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        content = scene_content(args.session, walls=not args.no_walls, octomap=not args.no_octomap)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from .ros2_node import run
    return run(content, frame_id=args.frame_id, live=args.live, server_ip=args.server_ip,
               rate_hz=args.rate, moveit=not args.no_moveit, ros_args=argv)


if __name__ == "__main__":
    sys.exit(main())
