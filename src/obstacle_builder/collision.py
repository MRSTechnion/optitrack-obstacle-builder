"""
Collision layer — the obstacles of an exported session as solid 3D objects.

Any planner (OMPL, a custom sampler, a drone or ground-robot controller…)
needs to ask "is my robot, at this pose, inside an obstacle?" and "can it
move straight from A to B?". This module answers that from a session's
scene.json + STLs, using FCL (the collision library behind MoveIt and most
OMPL setups):

    from obstacle_builder.collision import CollisionWorld, Sphere, Box

    world = CollisionWorld.from_session("session_20261112_103000")
    drone = Sphere(radius=0.15)

    world.collides(drone, (1.0, 0.5, 1.2))          # → bool
    world.distance(drone, (1.0, 0.5, 1.2))          # → metres (< 0: penetrating)
    world.segment_free((0, 0, 1), (2, 1, 1), 0.15)  # straight-line move, exact
    world.in_bounds((1.0, 0.5, 1.2))                # inside the room boundary
    world.bounds()                                  # → (lo xyz, hi xyz) for samplers

Frame: the session's world frame (Z-up, metres, OptiTrack origin).
Orientations: quaternion [qx, qy, qz, qw].

What is solid:
  - every obstacle (its convex hull, exactly as exported);
  - the boundary walls, if they were exported (walls=True);
  - optionally the floor, as everything below Z = 0 (floor=True) — useful for
    drones; leave it off for ground robots that rest on the floor.
The boundary floor polygon is *not* solid; use in_bounds() / bounds() for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fcl
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .scene.scene_writer import LoadedSession, load_session

logger = logging.getLogger(__name__)

# The optional solid floor: a slab whose top face is Z = 0
_FLOOR_SIZE_M      = 1000.0
_FLOOR_THICKNESS_M = 10.0


# ---------------------------------------------------------------------------
# Robot shapes (plain dataclasses — users never need to touch FCL)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sphere:
    """A sphere of the given radius, centred on the pose position."""
    radius: float


@dataclass(frozen=True)
class Box:
    """A box with full side lengths (x, y, z), centred on the pose position."""
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Cylinder:
    """A cylinder along its local Z axis, centred on the pose position."""
    radius: float
    length: float


@dataclass(frozen=True)
class Capsule:
    """A capsule (cylinder with hemispherical ends) along its local Z axis."""
    radius: float
    length: float   # of the cylindrical part


Shape = Sphere | Box | Cylinder | Capsule


def _fcl_geometry(shape: Shape):
    if isinstance(shape, Sphere):
        return fcl.Sphere(shape.radius)
    if isinstance(shape, Box):
        return fcl.Box(*shape.size)
    if isinstance(shape, Cylinder):
        return fcl.Cylinder(shape.radius, shape.length)
    if isinstance(shape, Capsule):
        return fcl.Capsule(shape.radius, shape.length)
    raise TypeError(f"Unsupported shape: {shape!r}")


def _transform(position, orientation=None) -> fcl.Transform:
    pos = np.asarray(position, dtype=float)
    if orientation is None:
        return fcl.Transform(pos)
    rot = Rotation.from_quat(np.asarray(orientation, dtype=float)).as_matrix()
    return fcl.Transform(rot, pos)


def _convex(vertices: np.ndarray) -> fcl.Convex:
    """FCL convex shape from a point set (its convex hull)."""
    hull = ConvexHull(vertices)
    faces = hull.simplices
    flat = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    return fcl.Convex(np.asarray(vertices, dtype=float), len(faces), flat)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class CollisionWorld:
    """The solid obstacles of a session, with collision and distance queries."""

    def __init__(self) -> None:
        self._objects: list[tuple[str, fcl.CollisionObject]] = []
        self._floor_xy: np.ndarray | None = None    # boundary polygon, CCW
        self._ceiling: float | None = None
        self._extent: tuple[np.ndarray, np.ndarray] | None = None   # of convex solids
        self._extents: list[tuple[np.ndarray, np.ndarray]] = []     # per convex solid

    # -- construction -------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: Path | str | LoadedSession,
        *,
        walls: bool = True,
        floor: bool = False,
    ) -> CollisionWorld:
        """
        Build the world from an exported session (directory, its scene.json,
        or an already loaded LoadedSession).

        walls : make the exported boundary walls solid (if any)
        floor : make everything below Z = 0 solid
        """
        s = session if isinstance(session, LoadedSession) else load_session(Path(session))
        world = cls()

        for body in s.bodies:
            world.add_convex(
                body.name, s.hulls[body.name].vertices,
                body.position.as_tuple(), body.orientation.as_tuple(),
            )

        b = s.boundary
        if b is not None:
            world._floor_xy = np.asarray(b.floor_vertices, dtype=float)[:, :2]
            world._ceiling = b.room_height_z if b.room_height_z > 0 else None
            if walls:
                for i, (verts, _faces) in enumerate(b.wall_meshes):
                    world.add_convex(f"boundary_wall_{i}", verts)

        if floor:
            # A slab under Z = 0, far larger than any capture volume. Not an
            # fcl.Halfspace: FCL's signed distance to a halfspace segfaults.
            world.add_object(
                "floor",
                fcl.Box(_FLOOR_SIZE_M, _FLOOR_SIZE_M, _FLOOR_THICKNESS_M),
                (0.0, 0.0, -_FLOOR_THICKNESS_M / 2),
            )
        return world

    def add_convex(self, name, vertices, position=(0, 0, 0), orientation=None) -> None:
        """Add a convex solid (the hull of `vertices`, in its local frame)."""
        verts = np.asarray(vertices, dtype=float)
        try:
            geom = _convex(verts)
        except QhullError as e:
            raise ValueError(f"'{name}': vertices do not span a 3D volume") from e
        self.add_object(name, geom, position, orientation)

        rot = Rotation.identity() if orientation is None else Rotation.from_quat(orientation)
        world_verts = rot.apply(verts) + np.asarray(position, dtype=float)
        lo, hi = world_verts.min(axis=0), world_verts.max(axis=0)
        self._extents.append((lo, hi))
        if self._extent is not None:
            lo, hi = np.minimum(lo, self._extent[0]), np.maximum(hi, self._extent[1])
        self._extent = (lo, hi)

    def add_object(self, name, geometry, position=(0, 0, 0), orientation=None) -> None:
        """Add any FCL geometry (advanced use)."""
        self._objects.append((name, fcl.CollisionObject(geometry, _transform(position, orientation))))

    def object_extents(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Axis-aligned (lo, hi) world extent of each convex solid (not the floor)."""
        return list(self._extents)

    @property
    def names(self) -> list[str]:
        """Names of the solid objects."""
        return [name for name, _ in self._objects]

    # -- queries -------------------------------------------------------------

    def colliding(self, shape: Shape, position, orientation=None) -> list[str]:
        """Names of the objects the shape at this pose touches or penetrates."""
        robot = fcl.CollisionObject(_fcl_geometry(shape), _transform(position, orientation))
        return self._colliding_with(robot)

    def collides(self, shape: Shape, position, orientation=None) -> bool:
        """True if the shape at this pose touches or penetrates any object."""
        robot = fcl.CollisionObject(_fcl_geometry(shape), _transform(position, orientation))
        req = fcl.CollisionRequest()
        return any(fcl.collide(robot, obj, req, fcl.CollisionResult()) for _, obj in self._objects)

    def point_in_obstacle(self, point) -> bool:
        """True if the point is inside (or on the surface of) any object."""
        return self.collides(Sphere(1e-6), point)

    def nearest(self, shape: Shape, position, orientation=None) -> tuple[str | None, float]:
        """
        (name, signed distance in metres) of the closest object. The distance
        is negative when penetrating. (None, inf) if the world is empty.
        """
        robot = fcl.CollisionObject(_fcl_geometry(shape), _transform(position, orientation))
        req = fcl.DistanceRequest(enable_signed_distance=True)
        best: tuple[str | None, float] = (None, float("inf"))
        for name, obj in self._objects:
            d = fcl.distance(robot, obj, req, fcl.DistanceResult())
            if d < best[1]:
                best = (name, float(d))
        return best

    def distance(self, shape: Shape, position, orientation=None) -> float:
        """Signed clearance to the closest object (m); negative = penetrating."""
        return self.nearest(shape, position, orientation)[1]

    def segment_free(self, start, end, radius: float) -> bool:
        """
        True if a sphere of `radius` can move in a straight line from `start`
        to `end` without touching any object — exact (the swept volume is a
        capsule), not sampled. Use it as an OMPL motion validator.
        """
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            return not self.collides(Sphere(radius), a)
        rot = Rotation.align_vectors([d / length], [[0.0, 0.0, 1.0]])[0]
        swept = fcl.CollisionObject(
            fcl.Capsule(radius, length), fcl.Transform(rot.as_matrix(), (a + b) / 2)
        )
        return not self._colliding_with(swept)

    # -- room bounds ------------------------------------------------------------

    @property
    def has_bounds(self) -> bool:
        """True if the session has a (Z-up) boundary."""
        return self._floor_xy is not None

    def in_bounds(self, point) -> bool:
        """
        True if the point is above the boundary floor polygon, between the
        floor (Z = 0) and the ceiling. True everywhere if there is no boundary.
        """
        return bool(self.in_bounds_many(np.asarray(point, dtype=float)[None, :])[0])

    def in_bounds_many(self, points) -> np.ndarray:
        """Vectorised in_bounds() for an (N, 3) array; returns (N,) booleans."""
        pts = np.asarray(points, dtype=float)
        if self._floor_xy is None:
            return np.ones(len(pts), dtype=bool)
        ok = pts[:, 2] >= 0.0
        if self._ceiling is not None:
            ok &= pts[:, 2] <= self._ceiling
        poly = self._floor_xy
        edges = np.roll(poly, -1, axis=0) - poly                # (E, 2)
        inside = np.empty(len(pts), dtype=bool)
        chunk = 100_000                                         # bounds memory: chunk x E temporaries
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            dx = p[:, None, 0] - poly[None, :, 0]               # (n, E)
            dy = p[:, None, 1] - poly[None, :, 1]
            cross = edges[None, :, 0] * dy - edges[None, :, 1] * dx
            inside[i:i + chunk] = np.all(cross >= -1e-9, axis=1)   # convex, CCW: left of every edge
        return ok & inside

    def bounds(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Axis-aligned (lo, hi) corners of the planning volume: the boundary
        polygon's extent, Z from 0 to the ceiling. Without a boundary: the
        obstacles' extent plus 1 m. `margin` shrinks it on every side.
        """
        if self._floor_xy is not None:
            lo = np.array([*self._floor_xy.min(axis=0), 0.0])
            hi = np.array([*self._floor_xy.max(axis=0), self._ceiling or 3.0])
        elif self._extent is not None:
            lo, hi = self._extent[0] - 1.0, self._extent[1] + 1.0
        else:
            raise ValueError("Empty world: no obstacles and no boundary.")
        return lo + margin, hi - margin

    # -- internals -----------------------------------------------------------

    def _colliding_with(self, robot: fcl.CollisionObject) -> list[str]:
        req = fcl.CollisionRequest()
        return [name for name, obj in self._objects
                if fcl.collide(robot, obj, req, fcl.CollisionResult())]

    def __repr__(self) -> str:
        return f"CollisionWorld({len(self._objects)} objects: {', '.join(self.names)})"
