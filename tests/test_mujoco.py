"""
Tests for the MuJoCo adapter (adapters/mujoco.py).

The scene is compiled with MuJoCo and checked against the collision layer:
MuJoCo's signed distance from a probe sphere to each obstacle must match
CollisionWorld's (both exact for convex shapes).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_builder.adapters import mujoco as adapter
from obstacle_builder.collision import CollisionWorld, Sphere
from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

mujoco = pytest.importorskip("mujoco")

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"

# Shaped like Crazyflow's crazyflow/scene.xml (meshdir points at drone assets)
_BASE_SCENE = """<mujoco model="Drone scene">
  <option integrator="RK4" timestep="0.001"/>
  <compiler inertiafromgeom="false" meshdir="assets" autolimits="true"/>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             width="64" height="64"/>
    <material name="groundplane" texture="groundplane"/>
  </asset>
  <worldbody>
    <!-- keep me -->
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def session(tmp_path):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    w = convert_frame(client.capture_frame())
    s = build_session(build_all_hulls(w.rigid_bodies),
                      build_boundary(w.cameras, wall_height_m=1.0), w.rigid_bodies, wall_height=1.0)
    return export_session(s, tmp_path / "s")


def _compile(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _probe_distances(path, points, radius):
    """MuJoCo signed distance from a sphere at each point to every named geom."""
    spec = mujoco.MjSpec.from_file(str(path))
    probe = spec.worldbody.add_body(name="probe", mocap=True)
    probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")
    fromto = np.zeros(6)
    out = []
    for p in points:
        data.mocap_pos[0] = p
        mujoco.mj_forward(model, data)
        out.append({
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g):
                mujoco.mj_geomDistance(model, data, probe_id, g, 10.0, fromto)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        })
    return out


def test_written_and_compiles(session):
    path = adapter.write_mjcf(session)
    assert path == session / "mujoco.xml"
    model, _ = _compile(path)
    loaded = load_session(session)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)}
    assert {b.name for b in loaded.bodies} <= names
    assert sum(n.startswith("boundary_wall_") for n in names if n) == len(loaded.boundary.wall_meshes)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    root = ET.parse(path).getroot()
    assert not any(m.get("file") for m in root.iter("mesh"))      # inline: no files to find


def test_bodies_at_scene_poses(session):
    model, data = _compile(adapter.write_mjcf(session))
    for body in load_session(session).bodies:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
        np.testing.assert_allclose(data.xpos[b], body.position.as_tuple(), atol=1e-6)
        got = Rotation.from_quat(np.roll(data.xquat[b], -1))      # MuJoCo w-first → x, y, z, w
        want = Rotation.from_quat(body.orientation.as_tuple())
        assert (got.inv() * want).magnitude() < 1e-5


def test_geometry_matches_collision_layer(session):
    """Signed distances agree with FCL inside, near and far from every obstacle and wall."""
    path = adapter.write_mjcf(session)
    world = CollisionWorld.from_session(session)
    radius = 0.05
    rng = np.random.default_rng(3)
    lo, hi = world.bounds()
    points = rng.uniform(lo + [0, 0, 0.1], hi, size=(60, 3))
    loaded = load_session(session)
    for body in loaded.bodies:
        v = loaded.hulls[body.name].vertices
        c = v.mean(axis=0)
        out = (v - c) / np.linalg.norm(v - c, axis=1, keepdims=True)
        local = np.vstack([c,                             # inside
                           v + out * (radius + 0.02),     # just clear of each vertex
                           v + out * (radius - 0.02)])    # just touching it
        world_pts = Rotation.from_quat(body.orientation.as_tuple()).apply(local) + body.position.as_tuple()
        points = np.vstack([points, world_pts])
    for p, dists in zip(points, _probe_distances(path, points, radius), strict=True):
        mj_min = min(dists.values())
        fcl_min = world.distance(Sphere(radius), p)
        if fcl_min > 0 or mj_min > 0:      # penetration depths are computed differently
            assert mj_min == pytest.approx(fcl_min, abs=2e-3), p
        assert (mj_min < 0) == world.collides(Sphere(radius), p), p


def test_base_scene_kept_and_obstacles_added(session, tmp_path):
    base = tmp_path / "crazyflow" / "scene.xml"
    base.parent.mkdir()
    base.write_text(_BASE_SCENE)
    out = adapter.write_mjcf(session, tmp_path / "lab.xml", base_scene=base)
    root = ET.parse(out, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    assert root.get("model") == "Drone scene"
    assert root.find("option").get("integrator") == "RK4"
    assert root.find("compiler").get("meshdir") == "assets"
    assert any("keep me" in (c.text or "") for c in root.iter(ET.Comment))
    assert len(root.find("worldbody").findall("geom")) == 1          # the base's own floor only
    model, _ = _compile(out)                     # inline meshes: meshdir does not matter
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert base.read_text() == _BASE_SCENE


def test_options(session, tmp_path):
    out = adapter.write_mjcf(session, tmp_path / "x.xml", walls=False, floor=False, model_name="lab")
    model, _ = _compile(out)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "boundary_wall_0") < 0
    assert ET.parse(out).getroot().get("model") == "lab"


@pytest.mark.parametrize("content", ["not xml <", "<sdf/>", "<mujoco><include file='a.xml'/></mujoco>"])
def test_bad_base_scene(session, tmp_path, content):
    bad = tmp_path / "bad.xml"
    bad.write_text(content)
    with pytest.raises(ValueError):
        adapter.write_mjcf(session, tmp_path / "o.xml", base_scene=bad)
    with pytest.raises(FileNotFoundError):
        adapter.write_mjcf(session, tmp_path / "o.xml", base_scene=tmp_path / "missing.xml")


def test_cli(session, tmp_path, capsys):
    out = tmp_path / "cli.xml"
    assert adapter.main([str(session), "-o", str(out)]) == 0
    assert out.exists() and "mujoco.viewer" in capsys.readouterr().out
    assert adapter.main([str(tmp_path / "missing")]) == 1
