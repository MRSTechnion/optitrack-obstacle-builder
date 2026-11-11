"""
Scene writer — assembles and exports a complete session.

Takes HullResult and BoundaryResult objects (already in world/Z-up frame)
and writes:

    <output_dir>/
        session_YYYYMMDD_HHMMSS/
            models/
                <name>.stl          one per rigid body
                boundary.stl        floor polygon
                boundary_wall_0.stl  )
                boundary_wall_1.stl  ) only if wall extrusion requested
                ...                  )
            scene.json

scene.json is the universal interface consumed by all downstream adapters
(Gazebo, ROS 2, Unity, etc.). See docs/scene_json_spec.md for the schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry.hull import BoundaryResult, HullResult, export_boundary_stl, export_hull_stl
from ..natnet.types import RigidBody

logger = logging.getLogger(__name__)

# scene.json schema version — bump when the format changes
_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Session dataclass — everything needed for one export
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    All the data for one export session.

    hulls        : dict mapping body name → HullResult
    boundary     : BoundaryResult (room boundary geometry), or None
    rigid_bodies : original RigidBody list (for metadata like marker_count)
    convention   : "z_up" or "y_up" — the coordinate convention used
    wall_height  : wall extrusion height in metres, or None
    wall_extrude_in : True = inward, False = outward
    notes        : optional free-text notes stored in scene.json
    """
    hulls:           dict[str, HullResult]
    boundary:        Optional[BoundaryResult]
    rigid_bodies:    list[RigidBody]
    convention:      str = "z_up"
    wall_height:     Optional[float] = None
    wall_extrude_in: bool = False
    notes:           str = ""


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_session(
    session:    Session,
    output_dir: Path,
) -> Path:
    """
    Export a complete session to disk.

    Creates a timestamped subdirectory under output_dir:
        output_dir/session_YYYYMMDD_HHMMSS/

    Returns the path of the session directory.

    Parameters
    ----------
    session    : Session object with all geometry results
    output_dir : parent directory chosen by the user via the GUI file dialog
    """
    # Create timestamped session folder
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir  = output_dir / f"session_{timestamp}"
    models_dir   = session_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting session to: {session_dir}")

    # --- Export STL files ---------------------------------------------------

    stl_records = []   # list of dicts — becomes the "models" array in scene.json

    # One STL per rigid body hull
    for name, hull in session.hulls.items():
        stl_path     = export_hull_stl(hull, models_dir)
        body         = _find_body(session.rigid_bodies, name)
        marker_count = body.marker_count if body else len(hull.vertices)

        # World-space pose comes from the original rigid body
        # (hull vertices are in body-local frame; pose places them in the world)
        position, quaternion = _get_world_pose(body)

        stl_records.append({
            "name":         name,
            "stl":          f"models/{stl_path.name}",
            "position":     _fmt_vec(position),
            "quaternion":   _fmt_quat(quaternion),
            "type":         "obstacle",
            "marker_count": marker_count,
            "hull_volume_m3":    round(hull.volume, 6),
            "dimensions_m": _fmt_vec(hull.dimensions),
        })

    # Boundary STL(s)
    boundary_record = None
    if session.boundary is not None:
        boundary_paths = export_boundary_stl(session.boundary, models_dir)
        b = session.boundary

        wall_stl_list = []
        if len(boundary_paths) > 1:
            wall_stl_list = [
                f"models/{p.name}"
                for p in boundary_paths[1:]   # first path is floor
            ]

        boundary_record = {
            "stl":          f"models/{boundary_paths[0].name}",
            "type":         "boundary",
            "position":     [0.0, 0.0, 0.0],
            "quaternion":   [0.0, 0.0, 0.0, 1.0],
            "room_dimensions": {
                "width_x":  round(b.room_width_x,  4),
                "depth_y":  round(b.room_depth_y,  4),
                "height_z": round(b.room_height_z, 4),
            },
            "wall_extrusion": (
                {
                    "height_m":    round(session.wall_height, 4),
                    "direction":   "inward" if session.wall_extrude_in else "outward",
                    "thickness_m": 0.02,
                    "stl_files":   wall_stl_list,
                }
                if session.wall_height else None
            ),
            "corner_camera_positions": [
                _fmt_vec(c.position.as_tuple())
                for c in b.corner_cameras
            ],
        }

    # --- Write scene.json ---------------------------------------------------

    scene = {
        "version":    _SCHEMA_VERSION,
        "convention": session.convention,
        "origin":     "optitrack",
        "units":      "metres",
        "created":    datetime.now().isoformat(timespec="seconds"),
        "notes":      session.notes,
        "models":     stl_records,
        "boundary":   boundary_record,
    }

    json_path = session_dir / "scene.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)

    logger.info(f"Wrote scene.json: {json_path}")
    logger.info(
        f"Session complete — "
        f"{len(stl_records)} obstacle(s), "
        f"boundary={'yes' if boundary_record else 'no'}"
    )

    return session_dir


# ---------------------------------------------------------------------------
# Session builder — convenience function used by the GUI
# ---------------------------------------------------------------------------

def build_session(
    hulls:           dict[str, HullResult],
    boundary:        Optional[BoundaryResult],
    rigid_bodies:    list[RigidBody],
    z_up:            bool = True,
    wall_height:     Optional[float] = None,
    wall_extrude_in: bool = False,
    notes:           str = "",
) -> Session:
    """
    Construct a Session from the geometry pipeline outputs.
    Called by the GUI just before export_session().

    Parameters
    ----------
    hulls           : from geometry.hull.build_all_hulls()
    boundary        : from geometry.hull.build_boundary(), or None
    rigid_bodies    : from frame.rigid_bodies (world frame)
    z_up            : True = Z-up convention, False = Y-up
    wall_height     : wall extrusion height in metres, or None
    wall_extrude_in : wall extrusion direction
    notes           : optional user notes
    """
    return Session(
        hulls           = hulls,
        boundary        = boundary,
        rigid_bodies    = rigid_bodies,
        convention      = "z_up" if z_up else "y_up",
        wall_height     = wall_height,
        wall_extrude_in = wall_extrude_in,
        notes           = notes,
    )


# ---------------------------------------------------------------------------
# scene.json loader — for downstream adapters and testing
# ---------------------------------------------------------------------------

def load_scene(scene_json_path: Path) -> dict:
    """
    Load a scene.json file and return its contents as a dict.

    Used by adapter scripts (Gazebo SDF generator, ROS 2 launcher, etc.)
    and by tests that verify exported output.

    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the file is not valid JSON or missing required keys.
    """
    if not scene_json_path.exists():
        raise FileNotFoundError(f"scene.json not found: {scene_json_path}")

    with open(scene_json_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid scene.json: {e}") from e

    _validate_scene(data)
    return data


def _validate_scene(data: dict) -> None:
    """
    Minimal validation of a loaded scene.json.
    Raises ValueError if required top-level keys are missing.
    """
    required = {"version", "convention", "units", "models"}
    missing  = required - data.keys()
    if missing:
        raise ValueError(
            f"scene.json missing required keys: {missing}"
        )

    for i, model in enumerate(data.get("models", [])):
        for key in ("name", "stl", "position", "quaternion"):
            if key not in model:
                raise ValueError(
                    f"Model entry {i} ('{model.get('name', '?')}') "
                    f"missing required key: '{key}'"
                )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_body(
    bodies: list[RigidBody],
    name:   str,
) -> Optional[RigidBody]:
    """Return the RigidBody with the given name, or None."""
    for b in bodies:
        if b.name == name:
            return b
    return None


def _get_world_pose(
    body: Optional[RigidBody],
) -> tuple[tuple, tuple]:
    """
    Extract world-space position and quaternion from a RigidBody.
    Returns identity pose if body is None (should not happen in practice).
    """
    if body is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    pos  = body.position.as_tuple()
    quat = body.orientation.as_tuple()
    return pos, quat


def _fmt_vec(v) -> list[float]:
    """Format a 3-element vector as a list of rounded floats."""
    return [round(float(x), 6) for x in v]


def _fmt_quat(q) -> list[float]:
    """Format a 4-element quaternion as a list of rounded floats."""
    return [round(float(x), 6) for x in q]