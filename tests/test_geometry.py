"""
Tests for the geometry pipeline using the sample fixture.

Run with:
    pytest tests/test_geometry.py -v

No OptiTrack hardware needed — all tests use the fixture file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from obstacle_builder.geometry.hull import (
    build_all_hulls,
    build_boundary,
    build_hull,
)
from obstacle_builder.geometry.transform import (
    convert_frame,
    convert_position,
    convert_quaternion,
    markers_to_local_frame,
)
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.natnet.types import Frame, Marker, Quaternion, RigidBody, Vector3
from obstacle_builder.scene.scene_writer import (
    build_session,
    export_session,
    load_scene,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_frame() -> Frame:
    """Load a Frame from the fixture file via MockNatNetConnection."""
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    frame = client.capture_frame()
    assert frame is not None
    return frame


@pytest.fixture
def world_frame(raw_frame) -> Frame:
    """Convert the fixture frame to Z-up world frame."""
    return convert_frame(raw_frame, z_up=True)


@pytest.fixture
def hulls(world_frame) -> dict:
    """Build hulls for all obstacle bodies in the fixture."""
    bodies = [b for b in world_frame.rigid_bodies if b.name.lower() != "boundary"]
    return build_all_hulls(bodies)


# ---------------------------------------------------------------------------
# NatNet types
# ---------------------------------------------------------------------------

class TestTypes:

    def test_vector3_as_tuple(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.as_tuple() == (1.0, 2.0, 3.0)

    def test_vector3_as_list(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.as_list() == [1.0, 2.0, 3.0]

    def test_quaternion_as_tuple(self):
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        assert q.as_tuple() == (0.0, 0.0, 0.0, 1.0)

    def test_rigid_body_marker_count(self, raw_frame):
        for body in raw_frame.rigid_bodies:
            assert body.marker_count == len(body.markers)
            assert body.marker_count >= 1

    def test_rigid_body_marker_positions(self, raw_frame):
        for body in raw_frame.rigid_bodies:
            positions = body.marker_positions
            assert len(positions) == body.marker_count
            for pos in positions:
                assert len(pos) == 3


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

class TestTransform:

    def test_zup_converts_y_to_z(self):
        """OptiTrack Y → world Z."""
        v = Vector3(0.0, 1.0, 0.0)   # straight up in OptiTrack
        result = convert_position(v, z_up=True)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(1.0)

    def test_zup_converts_z_to_neg_y(self):
        """OptiTrack Z → world -Y."""
        v = Vector3(0.0, 0.0, 1.0)
        result = convert_position(v, z_up=True)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(-1.0)
        assert result.z == pytest.approx(0.0)

    def test_zup_x_unchanged(self):
        """OptiTrack X → world X (unchanged)."""
        v = Vector3(1.0, 0.0, 0.0)
        result = convert_position(v, z_up=True)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_yup_passthrough(self):
        """Y-up mode returns coordinates unchanged."""
        v = Vector3(1.0, 2.0, 3.0)
        result = convert_position(v, z_up=False)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_identity_quaternion_unchanged(self):
        """Identity quaternion stays identity after conversion."""
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        result = convert_quaternion(q, z_up=True)
        # Quaternion norm must be preserved
        norm = (result.qx**2 + result.qy**2 + result.qz**2 + result.qw**2) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_convert_frame_body_count(self, raw_frame):
        """Converting a frame preserves the number of rigid bodies."""
        world = convert_frame(raw_frame, z_up=True)
        assert len(world.rigid_bodies) == len(raw_frame.rigid_bodies)

    def test_convert_frame_camera_count(self, raw_frame):
        """Converting a frame preserves the number of cameras."""
        world = convert_frame(raw_frame, z_up=True)
        assert len(world.cameras) == len(raw_frame.cameras)

    def test_convert_frame_marker_count_preserved(self, raw_frame):
        """Each body's marker count is unchanged after conversion."""
        world = convert_frame(raw_frame, z_up=True)
        for orig, conv in zip(raw_frame.rigid_bodies, world.rigid_bodies):
            assert conv.marker_count == orig.marker_count

    def test_markers_to_local_frame_centred(self):
        """
        Markers transformed to body-local frame should have
        centroid near the origin when the body position equals
        the centroid of the markers.
        """
        markers   = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                     (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
        body_pos  = (0.0, 0.0, 0.0)
        body_quat = (0.0, 0.0, 0.0, 1.0)   # identity

        local = markers_to_local_frame(markers, body_pos, body_quat)
        centroid = local.mean(axis=0)

        assert centroid == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)

    def test_markers_to_local_frame_shape(self):
        """Output shape is (N, 3) for N marker positions."""
        markers   = [(float(i), 0.0, 0.0) for i in range(8)]
        body_pos  = (0.0, 0.0, 0.0)
        body_quat = (0.0, 0.0, 0.0, 1.0)

        local = markers_to_local_frame(markers, body_pos, body_quat)
        assert local.shape == (8, 3)


# ---------------------------------------------------------------------------
# Hull building
# ---------------------------------------------------------------------------

class TestHull:

    def test_hulls_built_for_all_bodies(self, world_frame, hulls):
        """Every obstacle body in the fixture produces a hull."""
        body_names = {
            b.name for b in world_frame.rigid_bodies
            if b.name.lower() != "boundary"
        }
        assert set(hulls.keys()) == body_names

    def test_hull_has_valid_vertices(self, hulls):
        """Every hull has at least 4 vertices (minimum for a 3D convex hull)."""
        for name, hull in hulls.items():
            assert hull.vertices.shape[1] == 3, f"{name}: vertices not (N,3)"
            assert len(hull.vertices) >= 4, f"{name}: too few vertices"

    def test_hull_has_valid_faces(self, hulls):
        """Every hull has at least 4 triangular faces."""
        for name, hull in hulls.items():
            assert hull.faces.shape[1] == 3, f"{name}: faces not (M,3)"
            assert len(hull.faces) >= 4, f"{name}: too few faces"

    def test_hull_positive_volume(self, hulls):
        """Every hull has a positive volume."""
        for name, hull in hulls.items():
            assert hull.volume > 0, f"{name}: non-positive volume"

    def test_hull_face_indices_in_range(self, hulls):
        """Face indices must all be valid indices into the vertices array."""
        for name, hull in hulls.items():
            n_verts = len(hull.vertices)
            assert hull.faces.max() < n_verts, \
                f"{name}: face index {hull.faces.max()} out of range (n={n_verts})"
            assert hull.faces.min() >= 0, f"{name}: negative face index"

    def test_hull_dimensions_positive(self, hulls):
        """Bounding box dimensions must all be positive."""
        for name, hull in hulls.items():
            dims = hull.dimensions
            assert all(d > 0 for d in dims), \
                f"{name}: non-positive dimension {dims}"

    def test_too_few_markers_returns_none(self):
        """A body with fewer than 4 markers should not produce a hull."""
        body = RigidBody(
            body_id        = 99,
            name           = "sparse_body",
            position       = Vector3(0.0, 0.0, 0.0),
            orientation    = Quaternion(0.0, 0.0, 0.0, 1.0),
            markers        = [
                Marker(i, Vector3(float(i), 0.0, 0.0))
                for i in range(3)   # only 3 markers — insufficient
            ],
            tracking_valid = True,
            mean_error     = 0.0,
        )
        result = build_hull(body)
        assert result is None

    def test_invalid_tracking_returns_none(self):
        """A body with tracking_valid=False should not produce a hull."""
        body = RigidBody(
            body_id        = 100,
            name           = "lost_body",
            position       = Vector3(0.0, 0.0, 0.0),
            orientation    = Quaternion(0.0, 0.0, 0.0, 1.0),
            markers        = [
                Marker(i, Vector3(float(i), float(i), float(i)))
                for i in range(8)
            ],
            tracking_valid = False,   # lost tracking
            mean_error     = 0.02,
        )
        result = build_hull(body)
        assert result is None

    def test_coplanar_markers_returns_none(self):
        """Markers all on the same plane produce a degenerate hull → None."""
        body = RigidBody(
            body_id        = 101,
            name           = "flat_body",
            position       = Vector3(0.0, 0.0, 0.0),
            orientation    = Quaternion(0.0, 0.0, 0.0, 1.0),
            markers        = [
                Marker(0, Vector3(0.0, 0.0, 0.0)),
                Marker(1, Vector3(1.0, 0.0, 0.0)),
                Marker(2, Vector3(0.0, 1.0, 0.0)),
                Marker(3, Vector3(1.0, 1.0, 0.0)),
                Marker(4, Vector3(0.5, 0.5, 0.0)),
            ],
            tracking_valid = True,
            mean_error     = 0.0,
        )
        result = build_hull(body)
        assert result is None


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

class TestBoundary:

    def test_boundary_built_from_cameras(self, world_frame):
        """Boundary is built successfully when cameras are available."""
        boundary = build_boundary(world_frame.cameras)
        assert boundary is not None

    def test_boundary_has_4_floor_vertices(self, world_frame):
        """Floor polygon has exactly 4 vertices (rectangular room)."""
        boundary = build_boundary(world_frame.cameras)
        assert len(boundary.floor_vertices) == 4

    def test_boundary_floor_at_z_zero(self, world_frame):
        """All floor vertices should be at Z=0."""
        boundary = build_boundary(world_frame.cameras)
        for pt in boundary.floor_vertices:
            assert pt[2] == pytest.approx(0.0)

    def test_boundary_room_dimensions_positive(self, world_frame):
        """Room width and depth must be positive."""
        boundary = build_boundary(world_frame.cameras)
        assert boundary.room_width_x > 0
        assert boundary.room_depth_y > 0
        assert boundary.room_height_z > 0

    def test_boundary_corner_cameras(self, world_frame):
        """Exactly 4 corner cameras should be identified."""
        boundary = build_boundary(world_frame.cameras)
        assert len(boundary.corner_cameras) == 4

    def test_boundary_no_walls_by_default(self, world_frame):
        """Without wall_height, wall_meshes should be empty."""
        boundary = build_boundary(world_frame.cameras)
        assert boundary.wall_meshes == []

    def test_boundary_walls_generated(self, world_frame):
        """With wall_height set, 4 wall panels should be generated."""
        boundary = build_boundary(world_frame.cameras, wall_height_m=3.0)
        assert len(boundary.wall_meshes) == 4

    def test_boundary_walls_outward(self, world_frame):
        """Outward walls should not reduce the floor area."""
        b_no_walls = build_boundary(world_frame.cameras)
        b_outward  = build_boundary(
            world_frame.cameras,
            wall_height_m=3.0,
            wall_extrude_in=False,
        )
        # Floor vertices should be identical regardless of wall direction
        np.testing.assert_array_almost_equal(
            b_no_walls.floor_vertices,
            b_outward.floor_vertices,
        )

    def test_boundary_too_few_cameras(self):
        """Fewer than 4 cameras should return None."""
        from obstacle_builder.natnet.types import CameraInfo as CI
        real_cameras = [
            CI(camera_id=i, name=f"cam_{i}", position=Vector3(float(i), float(i), 3.0))
            for i in range(3)
        ]
        result = build_boundary(real_cameras)
        assert result is None


# ---------------------------------------------------------------------------
# Scene writer
# ---------------------------------------------------------------------------

class TestSceneWriter:

    def test_export_creates_session_dir(self, hulls, world_frame, tmp_path):
        """Export creates a timestamped session directory."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir = export_session(session, tmp_path)
        assert session_dir.exists()
        assert session_dir.is_dir()
        assert session_dir.name.startswith("session_")

    def test_export_creates_scene_json(self, hulls, world_frame, tmp_path):
        """Export creates a valid scene.json."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir = export_session(session, tmp_path)
        json_path   = session_dir / "scene.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert data["convention"] in ("z_up", "y_up")
        assert data["units"] == "metres"
        assert isinstance(data["models"], list)

    def test_export_creates_stl_files(self, hulls, world_frame, tmp_path):
        """One STL file per hull is created in models/."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir  = export_session(session, tmp_path)
        models_dir   = session_dir / "models"

        assert models_dir.exists()
        stl_files = list(models_dir.glob("*.stl"))
        assert len(stl_files) == len(hulls)

    def test_export_model_names_match_hulls(self, hulls, world_frame, tmp_path):
        """scene.json model names match the hull keys."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir = export_session(session, tmp_path)
        scene       = load_scene(session_dir / "scene.json")

        exported_names = {m["name"] for m in scene["models"]}
        assert exported_names == set(hulls.keys())

    def test_export_with_boundary(self, hulls, world_frame, tmp_path):
        """Export with boundary creates boundary.stl and records it in scene.json."""
        boundary = build_boundary(world_frame.cameras)
        session  = build_session(
            hulls        = hulls,
            boundary     = boundary,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir = export_session(session, tmp_path)
        scene       = load_scene(session_dir / "scene.json")

        assert scene["boundary"] is not None
        boundary_stl = session_dir / "models" / "boundary.stl"
        assert boundary_stl.exists()

    def test_load_scene_validates_required_keys(self, tmp_path):
        """load_scene raises ValueError for a scene.json missing required keys."""
        broken = {"version": "1.0", "units": "metres"}  # missing 'models', 'convention'
        json_path = tmp_path / "scene.json"
        with open(json_path, "w") as f:
            json.dump(broken, f)

        with pytest.raises(ValueError, match="missing required keys"):
            load_scene(json_path)

    def test_scene_json_positions_are_lists(self, hulls, world_frame, tmp_path):
        """Positions and quaternions in scene.json are plain lists of floats."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
        )
        session_dir = export_session(session, tmp_path)
        scene       = load_scene(session_dir / "scene.json")

        for model in scene["models"]:
            assert isinstance(model["position"],   list)
            assert isinstance(model["quaternion"], list)
            assert len(model["position"])   == 3
            assert len(model["quaternion"]) == 4

    def test_yup_convention_recorded(self, hulls, world_frame, tmp_path):
        """Y-up mode is recorded correctly in scene.json."""
        session = build_session(
            hulls        = hulls,
            boundary     = None,
            rigid_bodies = world_frame.rigid_bodies,
            z_up         = False,
        )
        session_dir = export_session(session, tmp_path)
        scene       = load_scene(session_dir / "scene.json")
        assert scene["convention"] == "y_up"