"""
Tests for the Gazebo adapter (adapters/gazebo.py).

Pure-Python checks always run. If Gazebo (`gz`) is installed, the generated
world is also validated with `gz sdf -k` and loaded headless with `gz sim`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_builder.adapters import gazebo
from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"


def _export(tmp_path, *, z_up=True, walls=True):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    world = convert_frame(client.capture_frame(), z_up=z_up)
    hulls = build_all_hulls(world.rigid_bodies)
    boundary = build_boundary(world.cameras, wall_height_m=1.0 if walls else None)
    session = build_session(hulls, boundary, world.rigid_bodies, z_up=z_up,
                            wall_height=1.0 if walls else None)
    return export_session(session, tmp_path)


@pytest.fixture
def session_dir(tmp_path):
    return _export(tmp_path)


def _models(world_path):
    root = ET.parse(world_path).getroot()
    return {m.get("name"): m for m in root.find("world").findall("model")}


def _pose(model):
    v = [float(x) for x in model.find("pose").text.split()]
    return np.array(v[:3]), Rotation.from_euler("xyz", v[3:])   # SDF: fixed-axis RPY


def test_world_written_next_to_scene(session_dir):
    path = gazebo.write_world(session_dir)
    assert path == session_dir / "world.sdf"
    root = ET.parse(path).getroot()
    assert root.tag == "sdf" and root.get("version") == gazebo.SDF_VERSION
    assert root.find("world").findall("plugin") == []      # generic: default systems


def test_models_static_with_mesh_visual_and_collision(session_dir):
    scene = json.loads((session_dir / "scene.json").read_text())
    models = _models(gazebo.write_world(session_dir))
    for m in scene["models"]:
        model = models[m["name"]]
        assert model.find("static").text == "true"
        link = model.find("link")
        assert link.find("collision/geometry/mesh/uri").text == m["stl"]
        assert link.find("visual/geometry/mesh/uri").text == m["stl"]


def test_obstacles_land_where_scene_json_puts_them(session_dir):
    """STL vertices moved by the SDF pose == moved by the scene.json pose."""
    # Arbitrary 3D orientations: single-axis rotations cannot tell a wrong
    # roll-pitch-yaw convention from the right one.
    scene_path = session_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    for i, m in enumerate(scene["models"]):
        m["quaternion"] = list(Rotation.random(random_state=i).as_quat())
    scene_path.write_text(json.dumps(scene))

    loaded = load_session(session_dir)
    models = _models(gazebo.write_world(session_dir))
    for body in loaded.bodies:
        verts = loaded.hulls[body.name].vertices
        expected = Rotation.from_quat(body.orientation.as_tuple()).apply(verts) + body.position.as_tuple()
        xyz, rot = _pose(models[body.name])
        np.testing.assert_allclose(rot.apply(verts) + xyz, expected, atol=1e-5)


def test_boundary_floor_is_visual_only_and_walls_are_solid(session_dir):
    models = _models(gazebo.write_world(session_dir))
    floor = models["boundary_floor"]
    assert floor.find("link/collision") is None
    assert float(floor.find("link/visual/transparency").text) > 0
    walls = [m for n, m in models.items() if n.startswith("boundary_wall_")]
    assert len(walls) == 4
    assert all(w.find("link/collision") is not None for w in walls)


def test_options_drop_ground_and_boundary(session_dir):
    models = _models(gazebo.write_world(session_dir, ground_plane=False, boundary=False))
    assert "ground_plane" not in models
    assert not any(n.startswith("boundary") for n in models)


def test_output_elsewhere_keeps_mesh_paths_valid(session_dir, tmp_path):
    out = tmp_path / "elsewhere" / "deep" / "my.sdf"
    gazebo.write_world(session_dir, out)
    for uri in ET.parse(out).getroot().iter("uri"):
        assert (out.parent / uri.text).resolve().exists(), uri.text


def test_names_are_sanitised_and_unique(session_dir):
    scene_path = session_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["models"][0]["name"] = "my box <1>"
    scene["models"][1]["name"] = "my box (1)"         # sanitises to the same name
    scene["models"][2]["name"] = "ground_plane"       # clashes with the ground model
    scene_path.write_text(json.dumps(scene))
    names = list(_models(gazebo.write_world(session_dir)))
    assert len(names) == len(set(names))
    assert "my_box_1" in names and "my_box_1_2" in names and "ground_plane_2" in names


def test_cli(session_dir, tmp_path, capsys):
    assert gazebo.main([str(session_dir / "scene.json"), "-o", str(tmp_path / "w.sdf")]) == 0
    assert (tmp_path / "w.sdf").exists()
    assert "gz sim" in capsys.readouterr().out
    assert gazebo.main([str(tmp_path / "missing")]) == 1


# ---------------------------------------------------------------------------
# With Gazebo installed: validate and load the world for real
# ---------------------------------------------------------------------------

needs_gz = pytest.mark.skipif(shutil.which("gz") is None, reason="Gazebo (gz) not installed")


@needs_gz
def test_gz_sdf_validates(session_dir):
    out = subprocess.run(["gz", "sdf", "-k", str(gazebo.write_world(session_dir))],
                         check=False, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "Valid" in out.stdout, out.stdout + out.stderr


@needs_gz
def test_gz_sim_loads_world_headless(session_dir, tmp_path):
    world = gazebo.write_world(session_dir)
    try:
        out = subprocess.run(
            ["gz", "sim", "-s", "-r", "--iterations", "50", "-v", "2", str(world)],
            check=False, capture_output=True, text=True, timeout=120,
            cwd=tmp_path,   # run from elsewhere: mesh paths must resolve via the world file
        )
    except subprocess.TimeoutExpired:
        pytest.skip("gz sim did not finish in time on this machine")
    log = out.stdout + out.stderr
    assert out.returncode == 0, log
    assert "[Err]" not in log, log
