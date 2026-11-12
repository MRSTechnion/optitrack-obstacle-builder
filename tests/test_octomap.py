"""
Tests for the OctoMap adapter (adapters/octomap.py).

Maps are read back with the OctoMap library (octomap-python). If g++ and the
system OctoMap headers are installed (liboctomap-dev, as used by ROS), the
file is also read by an independent C++ program and compared point by point.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import octomap
import pytest
from scipy.spatial import ConvexHull

from obstacle_builder.adapters import octomap as adapter
from obstacle_builder.collision import Box, CollisionWorld
from obstacle_builder.geometry.hull import HullResult, build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.natnet.types import CameraInfo, Marker, Quaternion, RigidBody, Vector3
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"


def _read(path, res):
    tree = octomap.OcTree(res)
    assert tree.readBinary(str(path).encode())
    return tree


def _state(tree, p) -> str:
    node = tree.search(np.asarray(p, dtype=float))
    try:
        return "occupied" if tree.isNodeOccupied(node) else "free"
    except octomap.NullPointerException:   # unknown space: no node
        return "unknown"


@pytest.fixture(scope="module")
def mock_session(tmp_path_factory):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    w = convert_frame(client.capture_frame())
    session = build_session(build_all_hulls(w.rigid_bodies),
                            build_boundary(w.cameras, wall_height_m=1.0), w.rigid_bodies, wall_height=1.0)
    return export_session(session, tmp_path_factory.mktemp("mock"))


def _cube_session(tmp_path, lo=(1.5, -0.5, 0.0), size=1.0, room=True):
    """One axis-aligned cube [lo, lo + size], optionally in a 6 x 4 x 3 m room."""
    pts = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)]) - size / 2
    hull = HullResult(vertices=pts, faces=ConvexHull(pts).simplices, volume=size ** 3,
                      centroid_offset=np.zeros(3), name="cube")
    centre = np.asarray(lo) + size / 2
    body = RigidBody(body_id=1, name="cube", position=Vector3(*centre), orientation=Quaternion(0, 0, 0, 1),
                     markers=[Marker(marker_id=i, position=Vector3(*p)) for i, p in enumerate(pts + centre)],
                     tracking_valid=True, mean_error=0.0)
    boundary = None
    if room:
        cams = [CameraInfo(camera_id=i, name=f"c{i}", position=Vector3(x, y, 3.0))
                for i, (x, y) in enumerate([(-3, -2), (3, -2), (3, 2), (-3, 2)])]
        boundary = build_boundary(cams)
    return export_session(build_session({"cube": hull}, boundary, [body]), tmp_path)


def test_written_next_to_scene_and_readable(mock_session):
    path = adapter.write_octomap(mock_session)
    assert path == mock_session / "octomap.bt"
    tree = _read(path, 0.05)
    assert tree.getResolution() == pytest.approx(0.05)
    assert tree.size() > 0


def test_every_obstacle_is_occupied(mock_session):
    tree = _read(adapter.write_octomap(mock_session), 0.05)
    loaded = load_session(mock_session)
    for body in loaded.bodies:
        # the hull vertices' mean is inside the convex hull
        from scipy.spatial.transform import Rotation
        inside = Rotation.from_quat(body.orientation.as_tuple()).apply(
            loaded.hulls[body.name].vertices.mean(axis=0)) + body.position.as_tuple()
        assert _state(tree, inside) == "occupied", body.name


def test_states_in_room(mock_session):
    tree = _read(adapter.write_octomap(mock_session), 0.05)
    assert _state(tree, (-2.0, 1.0, 1.5)) == "free"          # room air
    assert _state(tree, (-2.0, 1.0, -0.02)) == "occupied"    # floor layer
    assert _state(tree, (3.01, 0.0, 0.5)) == "occupied"      # exported wall panel
    assert _state(tree, (9.0, 9.0, 1.0)) == "unknown"        # outside the room
    assert _state(tree, (0.0, 0.0, 3.4)) == "unknown"        # above the ceiling


def test_exact_voxel_count_for_grid_aligned_cube(tmp_path):
    """
    A 1 m cube aligned to a 10 cm grid fills exactly 10x10x10 voxels: voxels
    that only touch its faces (zero overlap) are not occupied.
    """
    session = _cube_session(tmp_path, room=False)
    world = CollisionWorld.from_session(session)
    occ = adapter.occupied_voxels(world, 0.1)
    assert len(occ) == 1000
    np.testing.assert_allclose(occ.min(axis=0), [1.55, -0.45, 0.05])
    np.testing.assert_allclose(occ.max(axis=0), [2.45, 0.45, 0.95])
    tree = _read(adapter.write_octomap(session, resolution=0.1, floor=False, free_space=False), 0.1)
    assert _state(tree, (1.55, 0.0, 0.5)) == "occupied"      # first voxel inside the face
    assert _state(tree, (1.45, 0.0, 0.5)) == "unknown"       # voxel touching the face


def test_unaligned_obstacle_is_never_thinner(tmp_path):
    """Partly covered voxels count as occupied: the map contains the whole obstacle."""
    session = _cube_session(tmp_path, lo=(1.53, -0.47, 0.02), size=0.5, room=False)
    occ = adapter.occupied_voxels(CollisionWorld.from_session(session), 0.1)
    lo, hi = occ.min(axis=0) - 0.05, occ.max(axis=0) + 0.05          # voxel bounds
    assert np.all(lo <= [1.53, -0.47, 0.02]) and np.all(hi >= [2.03, 0.03, 0.52])
    assert len(occ) == 6 * 6 * 6                                       # 0.5 m spans 6 cells


@pytest.mark.parametrize(("kwargs", "point", "expected"), [
    ({"free_space": False}, (-2.0, 1.0, 1.5), "unknown"),
    ({"floor": False}, (-2.0, 1.0, -0.02), "unknown"),
    ({"walls": False}, (3.01, 0.0, 0.5), "unknown"),          # outside the room, no wall
])
def test_options(mock_session, tmp_path, kwargs, point, expected):
    out = adapter.write_octomap(mock_session, tmp_path / "m.bt", **kwargs)
    assert _state(_read(out, 0.05), point) == expected


def test_resolution_option(mock_session, tmp_path):
    out = adapter.write_octomap(mock_session, tmp_path / "coarse.bt", resolution=0.2)
    assert _read(out, 0.2).getResolution() == pytest.approx(0.2)
    with pytest.raises(ValueError):
        adapter.write_octomap(mock_session, tmp_path / "x.bt", resolution=0)


def test_no_boundary_session(tmp_path):
    session = _cube_session(tmp_path, room=False)
    tree = _read(adapter.write_octomap(session, resolution=0.1), 0.1)
    assert _state(tree, (2.0, 0.0, 0.5)) == "occupied"
    assert _state(tree, (5.0, 0.0, 0.5)) == "unknown"        # no room: no free space
    assert _state(tree, (2.0, 0.0, -0.05)) == "occupied"     # floor under the obstacles' extent


def test_cli(mock_session, tmp_path, capsys):
    out = tmp_path / "cli.bt"
    assert adapter.main([str(mock_session), "-o", str(out), "-r", "0.1"]) == 0
    assert out.exists() and "Wrote" in capsys.readouterr().out
    assert adapter.main([str(tmp_path / "missing")]) == 1


# ---------------------------------------------------------------------------
# Independent check: read the file with the system OctoMap C++ library
# ---------------------------------------------------------------------------

_CPP = r"""
#include <octomap/octomap.h>
#include <iostream>
int main(int argc, char** argv) {
  octomap::OcTree tree(0.1);
  if (argc < 2 || !tree.readBinary(argv[1])) return 1;
  double x, y, z;
  while (std::cin >> x >> y >> z) {
    octomap::OcTreeNode* n = tree.search(x, y, z);
    std::cout << (n == nullptr ? 0 : (tree.isNodeOccupied(n) ? 2 : 1)) << "\n";
  }
  return 0;
}
"""

needs_cpp_octomap = pytest.mark.skipif(
    shutil.which("g++") is None or not Path("/usr/include/octomap/octomap.h").exists(),
    reason="g++ and liboctomap-dev not installed",
)


@needs_cpp_octomap
def test_cpp_octomap_reader_agrees(mock_session, tmp_path):
    src, exe = tmp_path / "check.cpp", tmp_path / "check"
    src.write_text(_CPP)
    subprocess.run(["g++", "-O2", "-o", str(exe), str(src), "-loctomap", "-loctomath"],
                   check=True, capture_output=True)
    res = 0.05
    path = adapter.write_octomap(mock_session, tmp_path / "m.bt", resolution=res)
    world = CollisionWorld.from_session(mock_session)

    pts = np.random.default_rng(1).uniform([-3.5, -2.5, -0.2], [3.5, 2.5, 3.4], size=(5000, 3))
    out = subprocess.run([str(exe), str(path)], input="\n".join(f"{x} {y} {z}" for x, y, z in pts),
                         capture_output=True, text=True, check=True).stdout.split()
    got = np.array(out, dtype=int)

    side = res * (1 - 1e-6)
    voxel = Box((side, side, side))
    lo, hi = world.bounds()
    centres = (np.floor(pts / res) + 0.5) * res
    expected = np.array([
        2 if (world.collides(voxel, c)
              or (-res < c[2] < 0 and lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]))
        else 1 if world.in_bounds(c) else 0
        for c in centres
    ])
    assert (got == expected).all(), f"{(got != expected).sum()} mismatches"
    assert set(got) == {0, 1, 2}
