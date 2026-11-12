"""
Tests for the collision layer (obstacle_builder.collision).

A hand-made session with known geometry (a 1 m cube centred at (2, 0, 0.5),
a 10 x 8 m room) so every answer can be checked exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_builder.collision import Box, Capsule, CollisionWorld, Cylinder, Sphere
from obstacle_builder.geometry.hull import HullResult, build_boundary
from obstacle_builder.natnet.types import CameraInfo, Marker, Quaternion, RigidBody, Vector3
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

CUBE = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])


def _cube_hull(name="cube"):
    from scipy.spatial import ConvexHull
    h = ConvexHull(CUBE)
    return HullResult(vertices=CUBE.copy(), faces=h.simplices, volume=1.0,
                      centroid_offset=np.zeros(3), name=name)


def _body(name, pos, quat=(0, 0, 0, 1)):
    return RigidBody(body_id=1, name=name, position=Vector3(*pos), orientation=Quaternion(*quat),
                     markers=[Marker(marker_id=i, position=Vector3(*p)) for i, p in enumerate(CUBE)],
                     tracking_valid=True, mean_error=0.0)


def _room_cameras(w=10.0, d=8.0, h=3.0):
    return [CameraInfo(camera_id=i, name=f"c{i}", position=Vector3(x, y, h))
            for i, (x, y) in enumerate([(-w/2, -d/2), (w/2, -d/2), (w/2, d/2), (-w/2, d/2)])]


@pytest.fixture
def session_dir(tmp_path):
    hulls = {"cube": _cube_hull()}
    bodies = [_body("cube", (2.0, 0.0, 0.5))]
    boundary = build_boundary(_room_cameras(), wall_height_m=1.0)
    return export_session(build_session(hulls, boundary, bodies, wall_height=1.0), tmp_path)


@pytest.fixture
def world(session_dir):
    return CollisionWorld.from_session(session_dir)


def test_objects(world):
    assert world.names[0] == "cube"
    assert sum(n.startswith("boundary_wall_") for n in world.names) == 4


def test_sphere_collision_and_signed_distance(world):
    r = Sphere(0.2)
    assert not world.collides(r, (1.0, 0, 0.5))
    assert world.distance(r, (1.0, 0, 0.5)) == pytest.approx(0.3, abs=1e-6)   # face at x=1.5
    assert world.collides(r, (1.4, 0, 0.5))
    assert world.distance(r, (1.4, 0, 0.5)) == pytest.approx(-0.1, abs=1e-6)
    assert world.colliding(r, (1.4, 0, 0.5)) == ["cube"]
    assert world.nearest(r, (1.0, 0, 0.5))[0] == "cube"


def test_point_in_obstacle(world):
    assert world.point_in_obstacle((2.0, 0.0, 0.5))
    assert world.point_in_obstacle((2.49, 0.49, 0.99))
    assert not world.point_in_obstacle((2.51, 0.0, 0.5))


@pytest.mark.parametrize(("shape", "pos", "quat", "hits"), [
    (Box((0.4, 0.4, 0.4)), (1.25, 0, 0.5), None, False),            # face at 1.45
    (Box((0.4, 0.4, 0.4)), (1.35, 0, 0.5), None, True),
    # rotated 45 deg about Z: half-diagonal 0.283 reaches 1.25+0.283 > 1.5
    (Box((0.4, 0.4, 0.4)), (1.25, 0, 0.5), Rotation.from_euler("z", 45, degrees=True).as_quat(), True),
    (Cylinder(0.1, 2.0), (1.35, 0, 1.0), None, False),                 # radius 0.1: 1.45
    (Cylinder(0.1, 2.0), (1.45, 0, 1.0), None, True),
    # length = cylinder part only: lowest point = z - 0.5 - 0.1; cube top at 1.0
    (Capsule(0.1, 1.0), (2.0, 0, 1.55), None, True),                  # reaches 0.95
    (Capsule(0.1, 1.0), (2.0, 0, 1.65), None, False),                 # reaches 1.05
])
def test_other_robot_shapes(world, shape, pos, quat, hits):
    assert world.collides(shape, pos, quat) is hits


def test_segment_free_is_exact(world):
    assert not world.segment_free((0, 0, 0.5), (4, 0, 0.5), 0.1)          # through the cube
    assert world.segment_free((0, 1.0, 0.5), (4, 1.0, 0.5), 0.1)          # 0.5 m beside
    assert not world.segment_free((0, 0.65, 0.5), (4, 0.65, 0.5), 0.2)    # grazes (0.15 gap < 0.2)
    assert world.segment_free((0, 0, 1.5), (4, 0, 1.5), 0.2)              # over the top
    assert not world.segment_free((2, 0, 0.5), (2, 0, 0.5), 0.1)          # zero length, inside


def test_segment_free_agrees_with_dense_sampling(world):
    rng = np.random.default_rng(0)
    r = Sphere(0.15)
    for _ in range(40):
        a, b = rng.uniform([-1, -2, 0.1], [5, 2, 1.5], size=(2, 3))
        sampled_free = all(not world.collides(r, a + t * (b - a)) for t in np.linspace(0, 1, 400))
        exact_free = world.segment_free(a, b, 0.15)
        if exact_free:
            assert sampled_free        # exact says free → no sample may collide
        # (sampled_free but not exact_free can happen between samples: fine)


def test_bounds_and_in_bounds(world):
    lo, hi = world.bounds()
    np.testing.assert_allclose(lo, [-5, -4, 0])
    np.testing.assert_allclose(hi, [5, 4, 3])
    np.testing.assert_allclose(world.bounds(margin=0.5)[0], [-4.5, -3.5, 0.5])
    assert world.in_bounds((4.9, 3.9, 2.9))
    assert not world.in_bounds((5.1, 0, 1))
    assert not world.in_bounds((0, 0, -0.01))
    assert not world.in_bounds((0, 0, 3.1))


def test_walls_and_floor_options(session_dir):
    no_walls = CollisionWorld.from_session(session_dir, walls=False)
    assert no_walls.names == ["cube"]
    # wall panels extrude outward from the floor edge x=5 by 2 cm
    assert CollisionWorld.from_session(session_dir).collides(Sphere(0.05), (5.03, 0, 0.5))
    assert not no_walls.collides(Sphere(0.05), (5.03, 0, 0.5))
    with_floor = CollisionWorld.from_session(session_dir, floor=True)
    assert with_floor.collides(Sphere(0.1), (0, 0, 0.05))
    assert not with_floor.collides(Sphere(0.1), (0, 0, 0.15))


def test_floor_distance(session_dir):
    """Regression: distance() with floor=True used to segfault (FCL halfspace)."""
    world = CollisionWorld.from_session(session_dir, floor=True, walls=False)
    assert world.nearest(Sphere(0.3), (-3.0, 2.0, 1.0)) == ("floor", pytest.approx(0.7, abs=1e-6))
    assert world.distance(Sphere(0.3), (-3.0, 2.0, 0.2)) == pytest.approx(-0.1, abs=1e-6)
    for z in np.linspace(0.31, 2.5, 200):                      # many calls, no crash
        world.distance(Sphere(0.3), (-3.0, 2.0, z))


def test_rotated_obstacle(tmp_path):
    q = Rotation.from_euler("z", 45, degrees=True).as_quat()
    session = build_session({"cube": _cube_hull()}, None, [_body("cube", (0, 0, 0.5), q)])
    world = CollisionWorld.from_session(export_session(session, tmp_path))
    # rotated cube's corner reaches 0.707 along X; unrotated face would be at 0.5
    assert world.point_in_obstacle((0.65, 0, 0.5))
    assert not world.point_in_obstacle((0.72, 0, 0.5))
    assert not world.has_bounds
    lo, _hi = world.bounds()                         # obstacles' extent + 1 m
    assert lo[0] == pytest.approx(-0.707 - 1, abs=1e-3)


def test_accepts_loaded_session(session_dir):
    world = CollisionWorld.from_session(load_session(session_dir))
    assert "cube" in world.names


def test_empty_world():
    with pytest.raises(ValueError):
        CollisionWorld().bounds()
    assert CollisionWorld().nearest(Sphere(1), (0, 0, 0)) == (None, float("inf"))
