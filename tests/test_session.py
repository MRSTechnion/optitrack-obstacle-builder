"""
Round-trip tests for load_session(): export a session, load it back, and
compare with what was exported. No GUI, no hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"


@pytest.fixture
def exported(tmp_path):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    world = convert_frame(client.capture_frame(), z_up=True)
    hulls = build_all_hulls(world.rigid_bodies)
    boundary = build_boundary(world.cameras, wall_height_m=1.0)
    session = build_session(hulls, boundary, world.rigid_bodies, wall_height=1.0)
    return export_session(session, tmp_path), hulls, boundary, world


def _vertex_set(v):
    return {tuple(p) for p in np.round(v, 5)}


def test_hulls_and_poses_round_trip(exported):
    session_dir, hulls, _, world = exported
    loaded = load_session(session_dir)
    assert set(loaded.hulls) == set(hulls)
    for name, hull in hulls.items():
        assert _vertex_set(loaded.hulls[name].vertices) == _vertex_set(hull.vertices)
        assert len(loaded.hulls[name].faces) == len(hull.faces)
        assert loaded.hulls[name].volume == pytest.approx(hull.volume, abs=1e-6)
    poses = {b.name: b for b in loaded.bodies}
    for body in world.rigid_bodies:
        np.testing.assert_allclose(poses[body.name].position.as_tuple(), body.position.as_tuple(), atol=1e-6)
        np.testing.assert_allclose(poses[body.name].orientation.as_tuple(), body.orientation.as_tuple(), atol=1e-6)
        assert loaded.marker_counts[body.name] == body.marker_count


def test_boundary_round_trip(exported):
    session_dir, _, boundary, _ = exported
    b = load_session(session_dir / "scene.json").boundary      # scene.json path works too
    np.testing.assert_allclose(b.floor_vertices, boundary.floor_vertices, atol=1e-6)
    assert len(b.wall_meshes) == len(boundary.wall_meshes)
    assert len(b.corner_cameras) == 4
    assert b.mode == boundary.mode
    assert b.room_width_x == pytest.approx(boundary.room_width_x, abs=1e-4)


def test_boundary_without_floor_polygon_is_rejected(exported):
    session_dir = exported[0]
    scene_path = session_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    del scene["boundary"]["floor_polygon"]
    scene_path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="floor_polygon"):
        load_session(session_dir)

def test_missing_stl_is_reported(exported):
    session_dir, hulls, _, _ = exported
    name = next(iter(hulls))
    (session_dir / "models" / f"{name}.stl").unlink()
    with pytest.raises(FileNotFoundError, match=name):
        load_session(session_dir)


def test_only_z_up_sessions_are_accepted(exported):
    """Z-up is the standard; every reader goes through load_scene()."""
    session_dir = exported[0]
    scene_path = session_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["convention"] = "y_up"
    scene_path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="z_up"):
        load_session(session_dir)
