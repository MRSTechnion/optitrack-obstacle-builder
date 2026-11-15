"""Tests for setup reconstruction (reconstruct.py) — no Qt."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.natnet.types import Quaternion, Vector3
from obstacle_builder.reconstruct import (
    State,
    check_placement,
    describe_move,
    summary,
    targets_from_session,
)
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    w = convert_frame(client.capture_frame())
    s = build_session(build_all_hulls(w.rigid_bodies), build_boundary(w.cameras), w.rigid_bodies)
    return load_session(export_session(s, tmp_path_factory.mktemp("s")))


@pytest.fixture
def targets(session):
    return targets_from_session(session)


def _moved(body, dx=(0.0, 0.0, 0.0), yaw_deg=0.0, valid=True):
    """`body` shifted by dx and turned by yaw_deg about its own pivot."""
    q = (Rotation.from_euler("z", yaw_deg, degrees=True)
         * Rotation.from_quat(body.orientation.as_tuple())).as_quat()
    return replace(body, position=Vector3(*(np.add(body.position.as_tuple(), dx))),
                   orientation=Quaternion(*q), tracking_valid=valid)


def test_targets_are_the_saved_poses_and_hull_centres(session, targets):
    assert list(targets) == [b.name for b in session.bodies]
    for body in session.bodies:
        t = targets[body.name]
        assert t.position == body.position.as_tuple()
        np.testing.assert_allclose(t.centre, session.hulls[body.name].vertices.mean(axis=0))


def test_all_placed_at_the_saved_poses(session, targets):
    p = check_placement(targets, session.bodies)
    assert all(x.state is State.PLACED for x in p.values())
    assert all(x.distance == pytest.approx(0, abs=1e-9) for x in p.values())
    assert summary(p) == "All 3 in place"


def test_moved_obstacle_is_misplaced_and_told_where_to_go(session, targets):
    a, *rest = session.bodies
    p = check_placement(targets, [_moved(a, dx=(0.10, -0.03, 0.0)), *rest])
    assert p[a.name].state is State.MISPLACED
    np.testing.assert_allclose(p[a.name].move, (-0.10, 0.03, 0.0), atol=1e-9)   # back again
    assert p[a.name].angle_deg == pytest.approx(0, abs=1e-6)
    assert describe_move(p[a.name].move) == "-x 10 cm, +y 3 cm"
    assert all(p[b.name].state is State.PLACED for b in rest)
    assert summary(p) == "2 / 3 in place"


@pytest.mark.parametrize(("shift", "expected"), [(0.019, State.PLACED), (0.021, State.MISPLACED)])
def test_position_tolerance(session, targets, shift, expected):
    a = session.bodies[0]
    assert check_placement(targets, [_moved(a, dx=(shift, 0, 0))],
                           position_tol=0.02)[a.name].state is expected


def test_turned_obstacle(session, targets):
    a = session.bodies[0]
    p = check_placement(targets, [_moved(a, yaw_deg=5.0)], position_tol=1.0)[a.name]
    assert p.angle_deg == pytest.approx(5.0, abs=1e-6)
    assert p.state is State.MISPLACED                          # 5° > 2°
    p = check_placement(targets, [_moved(a, yaw_deg=5.0)], position_tol=1.0, angle_tol=6.0)[a.name]
    assert p.state is State.PLACED


def test_distance_is_measured_at_the_hull_centre(session, targets):
    """Turning about the pivot moves the hull centre unless they coincide."""
    a = session.bodies[0]
    t = targets[a.name]
    p = check_placement(targets, [_moved(a, yaw_deg=90.0)])[a.name]
    arm = Rotation.from_quat(t.orientation).apply(t.centre)   # pivot → centre, world axes
    r = np.hypot(*arm[:2])                 # centre's distance from the vertical through the pivot
    assert p.distance == pytest.approx(np.sqrt(2) * r, rel=1e-6)


def test_missing_untracked_and_extra_bodies(session, targets):
    a, b, c = session.bodies
    stranger = replace(a, name="not_in_session")
    p = check_placement(targets, [_moved(b, valid=False), c, stranger])
    assert p[a.name].state is State.MISSING and p[a.name].move is None
    assert p[b.name].state is State.MISSING                    # streamed but not tracked
    assert p[c.name].state is State.PLACED
    assert "not_in_session" not in p
    assert summary(p) == "1 / 3 in place"


def test_describe_move_ignores_millimetres():
    assert describe_move(np.array([0.001, -0.002, 0.004])) == "—"
    assert describe_move(np.array([0.0, 0.0, 0.25])) == "+z 25 cm"
