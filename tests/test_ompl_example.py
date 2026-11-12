"""
Tests for examples/ompl_planning.py. Skipped where OMPL is not installed
(e.g. Windows: OMPL has no Windows wheels).

Paths are verified independently of the planner, by densely sampling them
against the collision world.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("ompl")

from obstacle_builder.collision import Box, CollisionWorld, Sphere
from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.scene.scene_writer import build_session, export_session

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("ompl_planning", ROOT / "examples" / "ompl_planning.py")
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    client = MockNatNetConnection(str(ROOT / "tests" / "fixtures" / "sample_scene.json"))
    client.connect()
    w = convert_frame(client.capture_frame())
    session = build_session(build_all_hulls(w.rigid_bodies),
                            build_boundary(w.cameras, wall_height_m=1.0), w.rigid_bodies, wall_height=1.0)
    return export_session(session, tmp_path_factory.mktemp("s"))


def test_drone_path_is_collision_free(session_dir):
    world = CollisionWorld.from_session(session_dir, floor=True)
    start, goal, r = (-2.5, -1.5, 0.5), (2.5, 1.5, 0.5), 0.25
    assert not world.segment_free(start, goal, r)               # direct line is blocked
    path = example.plan_drone(world, start, goal, r, time_s=5.0)
    assert path is not None
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    robot = Sphere(r)
    for a, b in itertools.pairwise(path):
        a, b = np.array(a), np.array(b)
        for t in np.linspace(0, 1, 200):
            p = a + t * (b - a)
            # default margin 5 cm: allow 1 mm for floating point
            assert world.distance(robot, p) >= 0.05 - 1e-3, p
            assert world.in_bounds(p)


def test_ground_path_is_collision_free(session_dir):
    world = CollisionWorld.from_session(session_dir)
    size = (0.7, 0.31, 0.4)
    start, goal = (-2.5, -1.5, 0.0), (2.5, 1.5, 1.57)
    robot = Box(size)
    straight = [np.array(start[:2]) + t * (np.array(goal[:2]) - start[:2]) for t in np.linspace(0, 1, 200)]
    assert any(world.collides(robot, (x, y, size[2] / 2)) for x, y in straight)   # direct line blocked
    path = example.plan_ground(world, start, goal, size, time_s=5.0)
    assert path is not None
    for (x0, y0, a0), (x1, y1, a1) in itertools.pairwise(path):
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi              # SE2 interpolates the short way
        for t in np.linspace(0, 1, 200):
            x, y, yaw = x0 + t * (x1 - x0), y0 + t * (y1 - y0), a0 + t * da
            q = Rotation.from_euler("z", yaw).as_quat()
            # the 5 cm margin must absorb the gaps between OMPL's samples
            assert not world.collides(robot, (x, y, size[2] / 2), q), (x, y, yaw)
            assert world.distance(robot, (x, y, size[2] / 2), q) > 0.03, (x, y, yaw)


def test_invalid_start_is_reported(session_dir):
    world = CollisionWorld.from_session(session_dir, floor=True)
    inside = world.bounds()[0] - 1.0                             # outside the room
    with pytest.raises(ValueError, match="start"):
        example.plan_drone(world, inside, (2.5, 1.5, 0.5), 0.25, time_s=1.0)


def test_cli(session_dir, tmp_path):
    out = tmp_path / "path.json"
    code = example.main([str(session_dir), "--robot", "drone", "--start", "-2.5", "-1.5", "0.5",
                         "--goal", "2.5", "1.5", "0.5", "--output", str(out)])
    assert code == 0 and out.exists()
