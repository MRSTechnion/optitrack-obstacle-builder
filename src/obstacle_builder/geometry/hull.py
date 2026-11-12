"""
Geometry reconstruction from marker positions.

Pipeline per rigid body:
    1. Take world-frame marker positions (already converted by transform.py)
    2. Transform markers into body-local frame (centred at origin)
    3. Compute convex hull of local marker positions
    4. Export hull as STL (vertices centred at origin, body-local axes)

Room boundary pipeline:
    1. Take world-frame camera positions
    2. Auto-detect the 4 corner cameras (largest bounding rectangle)
    3. Project corners onto the floor plane (Z=0)
    4. Optionally extrude upward to create thin wall STLs
    5. Export boundary STL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from stl import mesh as stl_mesh  # numpy-stl

from ..natnet.types import CameraInfo, RigidBody
from .transform import markers_to_local_frame

logger = logging.getLogger(__name__)

# Minimum number of markers needed to compute a 3D convex hull (non-coplanar)
_MIN_MARKERS_FOR_HULL = 4

# Minimum volume of a valid hull in m³ — below this the hull is degenerate
# (all markers coplanar or nearly so)
_MIN_HULL_VOLUME_M3 = 1e-6

# Wall extrusion default thickness (metres) — thin enough to be a visual
# boundary without being physically significant
_WALL_THICKNESS_M = 0.02


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HullResult:
    """
    The reconstructed geometry for one rigid body.

    vertices : np.ndarray shape (N, 3) — unique vertices of the convex hull
               in body-local frame, centred at (0,0,0)
    faces    : np.ndarray shape (M, 3) — triangle face indices into vertices
    volume   : float — hull volume in m³ (useful for sanity checking)
    centroid_offset : np.ndarray shape (3,) — offset of the geometric centre
                      from the body origin in body-local frame.
                      Usually close to (0,0,0) but not exactly if markers
                      are asymmetrically placed.
    name     : str — rigid body name, becomes the STL filename
    """
    vertices:        np.ndarray
    faces:           np.ndarray
    volume:          float
    centroid_offset: np.ndarray
    name:            str

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box: (min_xyz, max_xyz) in body-local frame."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def dimensions(self) -> np.ndarray:
        """Bounding box size (dx, dy, dz) in metres."""
        lo, hi = self.bounds
        return hi - lo


@dataclass
class BoundaryResult:
    """
    The room boundary geometry.

    floor_vertices : np.ndarray shape (N, 3) — floor polygon at Z=0, CCW from
                     above (N = 4 in "corners" mode)
    floor_faces    : np.ndarray shape (N-2, 3) — triangles for the floor polygon
    wall_meshes    : list of (vertices, faces) per wall panel, or empty list
                     if wall extrusion was not requested
    corner_cameras : list of CameraInfo — the cameras at the polygon vertices
    room_width_x   : float — room width in metres (see build_boundary)
    room_depth_y   : float — room depth in metres (see build_boundary)
    room_height_z  : float — mean height of corner_cameras (ceiling) in metres
    mode           : "corners", "all" or "rigid_body" — how it was built
    """
    floor_vertices:  np.ndarray
    floor_faces:     np.ndarray
    wall_meshes:     list[tuple[np.ndarray, np.ndarray]]
    corner_cameras:  list[CameraInfo]
    room_width_x:    float
    room_depth_y:    float
    room_height_z:   float
    mode:            str = "corners"


# How the floor polygon is built from the cameras (see build_boundary)
BOUNDARY_MODES = ("corners", "all", "rigid_body")

# Name of the rigid body that marks an interest area ("rigid_body" mode).
# Matched case-insensitively; never treated as an obstacle.
BOUNDARY_BODY_NAME = "boundary"


# ---------------------------------------------------------------------------
# Rigid body → convex hull
# ---------------------------------------------------------------------------

def build_hull(body: RigidBody) -> HullResult | None:
    """
    Reconstruct the convex hull geometry for one rigid body from its
    marker positions.

    The body must already be in world (Z-up) frame — call
    transform.convert_frame() before passing it here.

    Returns None if the hull cannot be computed (too few markers,
    degenerate geometry, tracking invalid).
    """
    if not body.tracking_valid:
        logger.warning(
            f"'{body.name}': skipping hull — tracking invalid. "
            "Ensure all markers are visible to the cameras."
        )
        return None

    if body.marker_count < _MIN_MARKERS_FOR_HULL:
        logger.warning(
            f"'{body.name}': only {body.marker_count} markers — "
            f"need at least {_MIN_MARKERS_FOR_HULL} non-coplanar markers "
            "for a valid 3D hull. See docs/marker_placement_guide.md."
        )
        return None

    # Get marker positions in world frame
    world_positions = body.marker_positions   # list of (x, y, z)

    # Transform into body-local frame (centred at origin, body axes)
    body_pos  = body.position.as_tuple()
    body_quat = body.orientation.as_tuple()
    local_pts = markers_to_local_frame(world_positions, body_pos, body_quat)
    # local_pts: np.ndarray shape (N, 3)

    # Check for degenerate input (all points on a line)
    if _is_collinear(local_pts):
        logger.warning(
            f"'{body.name}': all markers appear collinear — "
            "cannot build a 3D hull. Add markers on other faces."
        )
        return None

    # Compute convex hull
    try:
        hull = ConvexHull(local_pts)
    except QhullError as e:
        logger.warning(f"'{body.name}': ConvexHull failed — {e}. "
                       "Markers may be coplanar or too close together.")
        return None

    # Validate volume
    if hull.volume < _MIN_HULL_VOLUME_M3:
        logger.warning(
            f"'{body.name}': hull volume {hull.volume:.2e} m³ is too small "
            "(markers likely coplanar). Add markers on a second face."
        )
        return None

    # Extract unique vertices and re-indexed faces
    vertices, faces = _extract_hull_mesh(local_pts, hull)

    # Geometric centre of the hull in local frame
    # (usually near origin, but offset if markers are asymmetric)
    centroid_offset = vertices.mean(axis=0)

    logger.info(
        f"'{body.name}': hull OK — "
        f"{len(vertices)} vertices, {len(faces)} faces, "
        f"volume={hull.volume*1e6:.1f} cm³, "
        f"dims={np.round(_dimensions(vertices)*100, 1)} cm"
    )

    return HullResult(
        vertices        = vertices,
        faces           = faces,
        volume          = hull.volume,
        centroid_offset = centroid_offset,
        name            = body.name,
    )


def build_all_hulls(bodies: list[RigidBody]) -> dict[str, HullResult]:
    """
    Build hulls for a list of rigid bodies.
    Returns a dict mapping body name → HullResult.
    Bodies that fail hull computation are skipped (with a warning logged).
    """
    results = {}
    for body in bodies:
        result = build_hull(body)
        if result is not None:
            results[body.name] = result
    return results


# ---------------------------------------------------------------------------
# Room boundary
# ---------------------------------------------------------------------------

def build_boundary(
    cameras:         list[CameraInfo],
    wall_height_m:   float | None = None,
    wall_extrude_in: bool = False,
    mode:            str = "corners",
    boundary_body:   RigidBody | None = None,
) -> BoundaryResult | None:
    """
    Build the room boundary (floor polygon at Z=0) from camera positions.

    Modes
    -----
    "corners" (default): a 4-sided floor from the 4 cameras enclosing the
        largest area — the physical corners of a rectangular truss, in any
        orientation. room_width_x / room_depth_y are the rectangle's side
        lengths. Needs >= 4 cameras.
    "all": an N-sided floor — the convex hull (outline) of all cameras.
        Makes no assumption about the room's shape (circular rig, fewer
        cameras, irregular layout). room_width_x / room_depth_y are the
        polygon's extents along X and Y. Needs >= 3 non-collinear cameras.
    "rigid_body": an interest area marked by a rigid body named "boundary"
        (BOUNDARY_BODY_NAME) with markers on the floor at its corners: the
        floor is the outline of those markers (4 markers → quadrilateral,
        more → polygon). Cameras only set room_height_z (their mean height,
        or 0 if there are none). Needs boundary_body with >= 3 visible,
        non-collinear markers.

    Parameters
    ----------
    cameras         : list of CameraInfo in world (Z-up) frame
    wall_height_m   : if not None, generate thin wall panels extruded from
                      the floor boundary up to this height (one per edge)
    wall_extrude_in : True  → extrude inward (toward room centre)
                      False → extrude outward (toward physical walls) [default]
    mode            : "corners", "all" or "rigid_body"
    boundary_body   : the "boundary" rigid body (world frame), for
                      mode="rigid_body"; ignored otherwise

    Returns None if there are too few cameras / markers for the chosen mode.
    """
    if mode not in BOUNDARY_MODES:
        raise ValueError(f"mode must be one of {BOUNDARY_MODES}, got {mode!r}")

    if mode == "rigid_body":
        if boundary_body is None:
            logger.warning(
                f"No '{BOUNDARY_BODY_NAME}' rigid body in the frame — publish it "
                "in Motive to use it as the interest area."
            )
            return None
        marker_xy = np.array([[m.position.x, m.position.y] for m in boundary_body.markers])
        idx = _outline_indices(marker_xy) if len(marker_xy) >= 3 else None
        if idx is None:
            logger.warning(
                f"'{boundary_body.name}' needs at least 3 visible, non-collinear "
                f"markers (has {len(marker_xy)})."
            )
            return None
        corners = []   # not camera-based
        floor_pts = np.array([[*marker_xy[i], 0.0] for i in idx])
    else:
        min_cameras = 4 if mode == "corners" else 3
        if len(cameras) < min_cameras:
            logger.warning(
                f"Only {len(cameras)} cameras available — need at least "
                f"{min_cameras} for the '{mode}' boundary."
            )
            return None

        if mode == "corners":
            corners = _find_corner_cameras(cameras)
            for cam in corners:
                cam.is_corner = True
        else:
            corners = _outline_cameras(cameras)
            if corners is None:
                logger.warning("Cameras are collinear — cannot build a floor polygon.")
                return None

        # Floor projection: use X, Y (Z-up frame). Z=0 is the floor.
        floor_pts = np.array([
            [c.position.x, c.position.y, 0.0]
            for c in corners
        ])

    # Sort into consistent winding order (counter-clockwise from above)
    floor_pts = _sort_polygon_ccw(floor_pts)

    # Triangulate the floor polygon (simple fan from first vertex)
    floor_faces = _triangulate_polygon(len(floor_pts))

    if mode == "corners":
        room_width_x, room_depth_y = _rectangle_dimensions(floor_pts)
    else:
        extent = floor_pts[:, :2].max(axis=0) - floor_pts[:, :2].min(axis=0)
        room_width_x, room_depth_y = float(extent[0]), float(extent[1])
    height_cams   = corners if corners else cameras
    room_height_z = float(np.mean([c.position.z for c in height_cams])) if height_cams else 0.0

    shape = {
        "corners":    "rectangle",
        "all":        f"{len(floor_pts)}-sided polygon",
        "rigid_body": f"interest area, {len(floor_pts)} sides",
    }[mode]
    logger.info(
        f"Boundary ({shape}): {room_width_x:.2f} m × {room_depth_y:.2f} m, "
        f"ceiling height {room_height_z:.2f} m"
    )

    # Optional wall panels
    wall_meshes = []
    if wall_height_m is not None and wall_height_m > 0:
        wall_meshes = _build_walls(floor_pts, wall_height_m, wall_extrude_in)
        direction   = "inward" if wall_extrude_in else "outward"
        logger.info(
            f"Generated {len(wall_meshes)} wall panels "
            f"(height={wall_height_m:.2f} m, "
            f"thickness={_WALL_THICKNESS_M:.3f} m, "
            f"direction={direction})"
        )

    return BoundaryResult(
        floor_vertices = floor_pts,
        floor_faces    = floor_faces,
        wall_meshes    = wall_meshes,
        corner_cameras = corners,
        room_width_x   = room_width_x,
        room_depth_y   = room_depth_y,
        room_height_z  = room_height_z,
        mode           = mode,
    )


def _rectangle_dimensions(floor_pts: np.ndarray) -> tuple[float, float]:
    """
    (width, depth) of a 4-corner floor, measured along its own sides (correct
    even if the truss is rotated relative to Motive's axes): width is the
    mean length of the two sides closer to the X axis, depth of the other two.
    """
    edges   = np.roll(floor_pts[:, :2], -1, axis=0) - floor_pts[:, :2]
    lengths = np.linalg.norm(edges, axis=1)
    x_like  = np.abs(edges[:, 0]) >= np.abs(edges[:, 1])
    if x_like.sum() == 2:
        return float(lengths[x_like].mean()), float(lengths[~x_like].mean())
    # rotated ~45 deg: no clear X/Y sides, use opposite-side pairs
    return float((lengths[0] + lengths[2]) / 2), float((lengths[1] + lengths[3]) / 2)


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------

def export_stl(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """
    Write a triangle mesh to an STL file.

    Parameters
    ----------
    vertices : np.ndarray shape (N, 3)
    faces    : np.ndarray shape (M, 3) — indices into vertices
    path     : output file path (parent directory must exist)
    """
    n_faces = len(faces)
    obstacle_mesh = stl_mesh.Mesh(np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype))

    for i, face in enumerate(faces):
        for j in range(3):
            obstacle_mesh.vectors[i][j] = vertices[face[j]]

    obstacle_mesh.save(str(path))
    logger.info(f"Saved STL: {path} ({n_faces} triangles)")


def export_hull_stl(result: HullResult, output_dir: Path) -> Path:
    """
    Export a HullResult to an STL file in output_dir.
    Returns the path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.name}.stl"
    export_stl(result.vertices, result.faces, path)
    return path


def export_boundary_stl(result: BoundaryResult, output_dir: Path) -> list[Path]:
    """
    Export the boundary floor polygon and any wall panels to STL files.
    Returns list of paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Floor
    floor_path = output_dir / "boundary.stl"
    export_stl(result.floor_vertices, result.floor_faces, floor_path)
    written.append(floor_path)

    # Walls (one STL per panel)
    for i, (verts, faces) in enumerate(result.wall_meshes):
        wall_path = output_dir / f"boundary_wall_{i}.stl"
        export_stl(verts, faces, wall_path)
        written.append(wall_path)

    return written


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _extract_hull_mesh(
    points: np.ndarray,
    hull:   ConvexHull,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a clean (vertices, faces) mesh from a scipy ConvexHull.

    scipy's hull.simplices indexes into the original points array and
    may reference interior points. We re-index to only the hull vertices.
    """
    # hull.vertices: indices of points that are on the convex hull surface
    hull_vertex_indices = hull.vertices
    vertices = points[hull_vertex_indices]

    # hull.simplices: each row is a triangle as indices into points[]
    # Remap to index into our new vertices array
    old_to_new = {old: new for new, old in enumerate(hull_vertex_indices)}
    faces = np.array([
        [old_to_new[i] for i in simplex]
        for simplex in hull.simplices
    ])

    # Ensure consistent outward-facing normals using hull equations
    faces = _fix_winding(vertices, faces, hull)

    return vertices, faces


def _fix_winding(
    vertices: np.ndarray,
    faces:    np.ndarray,
    hull:     ConvexHull,
) -> np.ndarray:
    """
    Ensure all triangle normals point outward from the hull centroid.
    scipy does not guarantee consistent winding order.
    """
    centroid = vertices.mean(axis=0)
    fixed = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        centre_to_face = v0 - centroid
        if np.dot(normal, centre_to_face) < 0:
            fixed.append([face[0], face[2], face[1]])   # flip winding
        else:
            fixed.append(list(face))
    return np.array(fixed)


def _is_collinear(pts: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if all points lie on a single line."""
    if len(pts) < 3:
        return True
    v0 = pts[1] - pts[0]
    for p in pts[2:]:
        cross = np.cross(v0, p - pts[0])
        if np.linalg.norm(cross) > tol:
            return False
    return True


def _dimensions(vertices: np.ndarray) -> np.ndarray:
    """Return bounding box dimensions (dx, dy, dz) in metres."""
    return vertices.max(axis=0) - vertices.min(axis=0)


def _find_corner_cameras(cameras: list[CameraInfo]) -> list[CameraInfo]:
    """
    Find the 4 cameras that form the corners of the capture volume.

    Strategy: project all cameras onto the floor plane (X, Y) and pick the
    4 cameras that enclose the largest area. For a rectangular truss these
    are the physical corners, whether or not the truss is aligned with
    Motive's axes. (Taking min/max X and Y instead picks whichever side
    camera sticks out furthest - usually a mid-side camera, not a corner.)

    If cameras are already flagged is_corner=True (e.g. from fixture data),
    those are returned directly.
    """
    # Use pre-flagged corners if available (fixture or previous detection)
    pre_flagged = [c for c in cameras if c.is_corner]
    if len(pre_flagged) == 4:
        return pre_flagged

    floor_positions = np.array([
        [c.position.x, c.position.y]
        for c in cameras
    ])
    return [cameras[i] for i in _four_largest_hull_cameras(floor_positions)]


def _four_largest_hull_cameras(
    floor_positions: np.ndarray,
) -> list[int]:
    """
    Return the indices of the 4 points enclosing the largest quadrilateral.

    The optimal 4 always lie on the convex hull of all points, so only hull
    vertices are searched (a handful, even with many cameras).
    """
    from itertools import combinations

    try:
        candidates = list(ConvexHull(floor_positions).vertices)
    except QhullError:
        candidates = list(range(len(floor_positions)))  # degenerate layout
    if len(candidates) < 4:
        candidates = list(range(len(floor_positions)))

    best_area  = -1.0
    best_combo = candidates[:4]
    for combo in combinations(candidates, 4):
        try:
            h = ConvexHull(floor_positions[list(combo)])
        except QhullError:
            continue  # degenerate quad (collinear/duplicate cameras) - skip it
        if h.volume > best_area:   # ConvexHull.volume = area in 2D
            best_area  = h.volume
            best_combo = list(combo)

    return [int(i) for i in best_combo]


def _outline_indices(xy: np.ndarray) -> list[int] | None:
    """Indices of the 2D points on their convex hull; None if collinear."""
    try:
        return [int(i) for i in ConvexHull(xy).vertices]
    except QhullError:
        return None


def _outline_cameras(cameras: list[CameraInfo]) -> list[CameraInfo] | None:
    """
    Cameras on the convex hull (outline) of the floor projection, i.e. the
    vertices of the "all cameras" floor polygon. None if they are collinear.
    """
    idx = _outline_indices(np.array([[c.position.x, c.position.y] for c in cameras]))
    return None if idx is None else [cameras[i] for i in idx]


def _sort_polygon_ccw(pts: np.ndarray) -> np.ndarray:
    """
    Sort the vertices of a convex floor polygon into counter-clockwise
    winding order (when viewed from above, i.e. from +Z).
    Uses the centroid angle approach.
    """
    centroid = pts[:, :2].mean(axis=0)
    angles   = np.arctan2(
        pts[:, 1] - centroid[1],
        pts[:, 0] - centroid[0],
    )
    order = np.argsort(angles)
    return pts[order]


def _triangulate_polygon(n_vertices: int) -> np.ndarray:
    """
    Fan triangulation of a convex polygon with n_vertices.
    Returns face indices as (N-2, 3) array.
    Triangle fan from vertex 0: (0,1,2), (0,2,3), (0,3,4), ...
    """
    faces = []
    for i in range(1, n_vertices - 1):
        faces.append([0, i, i + 1])
    return np.array(faces)


def _build_walls(
    floor_pts:       np.ndarray,
    wall_height:     float,
    extrude_inward:  bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build thin wall panels from the floor boundary polygon.

    Each wall is a rectangular panel between two adjacent floor corners,
    extruded upward by wall_height. The panel has _WALL_THICKNESS_M depth.

    extrude_inward=False (default) → panels grow outward toward physical walls.
    extrude_inward=True            → panels grow inward into the experiment area.

    Returns a list of (vertices, faces) tuples, one per wall panel.
    """
    n = len(floor_pts)
    centroid_xy = floor_pts[:, :2].mean(axis=0)
    walls = []

    for i in range(n):
        p0 = floor_pts[i]
        p1 = floor_pts[(i + 1) % n]

        # Edge direction and perpendicular normal
        edge       = p1[:2] - p0[:2]
        edge_len   = np.linalg.norm(edge)
        if edge_len < 1e-6:
            continue

        edge_unit  = edge / edge_len
        inward     = centroid_xy - p0[:2]
        normal_ccw = np.array([-edge_unit[1], edge_unit[0]])

        # Make normal_ccw point inward first, then flip if outward was requested
        if np.dot(normal_ccw, inward) < 0:
            normal_ccw = -normal_ccw                    # now pointing inward
        if not extrude_inward:
            normal_ccw = -normal_ccw                    # flip to outward

        extrude_3d = np.array([
            normal_ccw[0] * _WALL_THICKNESS_M,
            normal_ccw[1] * _WALL_THICKNESS_M,
            0.0,
        ])

        # 8 vertices of the wall box
        # Outer face (at floor edge): p0_bot, p1_bot, p1_top, p0_top
        # Inner face (inward offset):  p0_bot_in, p1_bot_in, ...
        p0_bot     = p0.copy();             p0_bot[2]     = 0.0
        p1_bot     = p1.copy();             p1_bot[2]     = 0.0
        p0_top     = p0.copy();             p0_top[2]     = wall_height
        p1_top     = p1.copy();             p1_top[2]     = wall_height
        p0_bot_in  = p0_bot  + extrude_3d
        p1_bot_in  = p1_bot  + extrude_3d
        p0_top_in  = p0_top  + extrude_3d
        p1_top_in  = p1_top  + extrude_3d

        verts = np.array([
            p0_bot, p1_bot, p1_top, p0_top,       # 0-3: outer face
            p0_bot_in, p1_bot_in, p1_top_in, p0_top_in,  # 4-7: inner face
        ])

        # 6 faces × 2 triangles each = 12 triangles total
        faces = np.array([
            # Outer face
            [0, 1, 2], [0, 2, 3],
            # Inner face (reversed winding = inward-facing)
            [4, 6, 5], [4, 7, 6],
            # Bottom cap
            [0, 4, 1], [1, 4, 5],
            # Top cap
            [3, 2, 7], [2, 6, 7],
            # Left cap
            [0, 3, 4], [3, 7, 4],
            # Right cap
            [1, 5, 2], [2, 5, 6],
        ])

        walls.append((verts, faces))

    return walls