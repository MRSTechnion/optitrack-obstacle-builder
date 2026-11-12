"""
Tests for room-boundary detection from camera positions.

The lab layout below is the real 20-camera truss (world Z-up, metres), read
from Motive on 2026-11-12. The old min/max-X/Y corner detection picked
mid-side cameras there instead of the corners.
"""

from __future__ import annotations

import numpy as np
import pytest

from obstacle_builder.geometry.hull import build_boundary
from obstacle_builder.natnet.types import CameraInfo, Marker, Quaternion, RigidBody, Vector3

LAB_CAMERAS = [
    (-5.242, 0.502, 3.197),
    (-5.377, -1.733, 2.882),
    (1.124, 4.875, 3.203),
    (5.174, -5.097, 3.133),
    (5.106, -1.219, 3.150),
    (-5.431, 4.658, 3.256),
    (5.041, 4.966, 3.190),
    (-0.482, -5.234, 3.144),
    (-5.122, -5.327, 3.141),
    (5.062, 0.919, 3.195),
    (3.147, 5.049, 2.922),
    (-0.953, 5.000, 2.944),
    (-2.716, 4.789, 3.211),
    (5.149, 3.145, 2.945),
    (1.664, -5.274, 2.909),
    (-2.482, -5.396, 2.886),
    (-5.498, 2.324, 2.915),
    (3.635, -5.248, 2.894),
    (5.305, -2.988, 2.857),
    (-5.170, -3.543, 3.144),
]

LAB_CORNERS = {(-5.431, 4.658), (-5.122, -5.327), (5.174, -5.097), (5.041, 4.966)}


def _cams(points):
    return [
        CameraInfo(camera_id=i, name=f"cam_{i}", position=Vector3(*p))
        for i, p in enumerate(points)
    ]


def _corner_set(boundary):
    return {(round(c.position.x, 3), round(c.position.y, 3)) for c in boundary.corner_cameras}


def test_lab_layout_picks_physical_corners():
    b = build_boundary(_cams(LAB_CAMERAS))
    assert _corner_set(b) == LAB_CORNERS


def test_lab_layout_room_dimensions():
    b = build_boundary(_cams(LAB_CAMERAS))
    assert b.room_width_x == pytest.approx(10.39, abs=0.01)
    assert b.room_depth_y == pytest.approx(10.03, abs=0.01)
    assert b.room_height_z == pytest.approx(3.18, abs=0.01)


def _truss(width, depth, angle_deg, bulge=0.3):
    """Rectangular truss: 4 corners + 2 cameras per side bulging outward."""
    hw, hd = width / 2, depth / 2
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    corners = list(local)
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1], strict=True):
        n = np.array([y1 - y0, x0 - x1]) / np.hypot(x1 - x0, y1 - y0)  # outward normal
        for t in (1 / 3, 2 / 3):
            local.append(tuple(np.array([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]) + bulge * n))
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = [(*(rot @ np.array(p)), 3.0) for p in local]
    return pts, {tuple(np.round(rot @ np.array(c), 3)) for c in corners}


@pytest.mark.parametrize("angle", [0, 17, 45, -30])
def test_rotated_truss_with_bulging_side_cameras(angle):
    """Side cameras sticking out past the corners must not be picked."""
    pts, expected = _truss(10.0, 8.0, angle)
    b = build_boundary(_cams(pts))
    assert _corner_set(b) == expected
    assert sorted([b.room_width_x, b.room_depth_y]) == pytest.approx([8.0, 10.0], abs=1e-6)


# ---------------------------------------------------------------------------
# "all" mode: N-sided polygon from the outline of all cameras
# ---------------------------------------------------------------------------

def _ring(n, radius=4.0, z=3.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(radius * np.cos(t), radius * np.sin(t), z) for t in a]


def test_all_mode_circular_rig_gives_n_gon():
    b = build_boundary(_cams(_ring(12)), mode="all")
    assert b.mode == "all"
    assert len(b.floor_vertices) == 12
    assert len(b.floor_faces) == 10                      # fan: N-2 triangles
    assert np.allclose(b.floor_vertices[:, 2], 0.0)
    assert b.room_width_x == pytest.approx(8.0, abs=1e-6)


def test_all_mode_floor_is_ccw_from_above():
    b = build_boundary(_cams(_ring(9)), mode="all")
    xy = b.floor_vertices[:, :2]
    signed_area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    assert signed_area > 0


def test_all_mode_ignores_interior_cameras():
    pts = _ring(8) + [(0.5, -0.3, 3.0), (-1.0, 1.0, 2.5)]   # two cameras inside the ring
    b = build_boundary(_cams(pts), mode="all")
    assert len(b.floor_vertices) == 8


def test_all_mode_lab_layout_contains_every_camera():
    b = build_boundary(_cams(LAB_CAMERAS), mode="all")
    poly = b.floor_vertices[:, :2]
    edges = np.roll(poly, -1, axis=0) - poly
    for x, y, _ in LAB_CAMERAS:
        # inside (or on) a CCW convex polygon: left of / on every edge
        cross = edges[:, 0] * (y - poly[:, 1]) - edges[:, 1] * (x - poly[:, 0])
        assert np.all(cross >= -1e-9)
    # the lab outline has more than 4 vertices (side cameras bulge slightly)
    assert len(poly) > 4


def test_all_mode_works_with_three_cameras_corners_does_not():
    pts = _ring(3)
    assert build_boundary(_cams(pts), mode="corners") is None
    assert len(build_boundary(_cams(pts), mode="all").floor_vertices) == 3


def test_all_mode_one_wall_per_edge():
    b = build_boundary(_cams(_ring(7)), mode="all", wall_height_m=1.0)
    assert len(b.wall_meshes) == 7


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        build_boundary(_cams(LAB_CAMERAS), mode="circle")


@pytest.mark.parametrize(("mode", "json_mode", "n"), [("corners", "corners", 4), ("all", "all_cameras", 12)])
def test_scene_json_records_mode_and_floor_polygon(tmp_path, mode, json_mode, n):
    import json

    from obstacle_builder.scene.scene_writer import build_session, export_session

    pts = _ring(12) if mode == "all" else LAB_CAMERAS
    b = build_boundary(_cams(pts), mode=mode, wall_height_m=1.0)
    session_dir = export_session(build_session({}, b, [], wall_height=1.0), tmp_path)
    rec = json.loads((session_dir / "scene.json").read_text())["boundary"]

    assert rec["mode"] == json_mode
    assert len(rec["floor_polygon"]) == n
    assert len(rec["wall_extrusion"]["stl_files"]) == n
    assert (session_dir / "models" / "boundary.stl").exists()


# ---------------------------------------------------------------------------
# "rigid_body" mode: interest area marked by a rigid body named "boundary"
# ---------------------------------------------------------------------------


def _boundary_body(xy, z=0.01, name="boundary"):
    return RigidBody(
        body_id=99, name=name,
        position=Vector3(0.0, 0.0, z), orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        markers=[Marker(marker_id=i, position=Vector3(x, y, z)) for i, (x, y) in enumerate(xy)],
        tracking_valid=True, mean_error=0.0,
    )


AREA = [(-1.0, -0.5), (2.0, -0.5), (2.0, 1.5), (-1.0, 1.5)]   # 3 x 2 m


def test_rigid_body_mode_uses_marker_outline():
    b = build_boundary(_cams(LAB_CAMERAS), mode="rigid_body", boundary_body=_boundary_body(AREA))
    assert b.mode == "rigid_body"
    assert {tuple(v) for v in np.round(b.floor_vertices, 6)} == {(x, y, 0.0) for x, y in AREA}
    assert b.corner_cameras == []
    assert b.room_width_x == pytest.approx(3.0)
    assert b.room_depth_y == pytest.approx(2.0)
    # ceiling = mean height of all cameras (the capture volume), not the markers
    assert b.room_height_z == pytest.approx(np.mean([p[2] for p in LAB_CAMERAS]))


def test_rigid_body_mode_polygon_and_interior_markers():
    hexagon = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    body = _boundary_body(hexagon + [(0.1, 0.2)])     # one marker inside the area
    b = build_boundary([], mode="rigid_body", boundary_body=body, wall_height_m=0.5)
    assert len(b.floor_vertices) == 6
    assert len(b.wall_meshes) == 6
    assert b.room_height_z == 0.0                      # no cameras available


def test_rigid_body_mode_without_body_or_markers():
    assert build_boundary(_cams(LAB_CAMERAS), mode="rigid_body") is None
    two = _boundary_body(AREA[:2])
    assert build_boundary(_cams(LAB_CAMERAS), mode="rigid_body", boundary_body=two) is None
    line = _boundary_body([(0, 0), (1, 0), (2, 0)])
    assert build_boundary(_cams(LAB_CAMERAS), mode="rigid_body", boundary_body=line) is None


def test_rigid_body_mode_in_scene_json(tmp_path):
    import json

    from obstacle_builder.scene.scene_writer import build_session, export_session

    b = build_boundary(_cams(LAB_CAMERAS), mode="rigid_body", boundary_body=_boundary_body(AREA))
    session_dir = export_session(build_session({}, b, []), tmp_path)
    rec = json.loads((session_dir / "scene.json").read_text())["boundary"]
    assert rec["mode"] == "rigid_body"
    assert len(rec["floor_polygon"]) == 4
    assert rec["corner_camera_positions"] == []
