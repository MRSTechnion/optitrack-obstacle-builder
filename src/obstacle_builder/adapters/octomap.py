"""
OctoMap adapter — turn an exported session into a 3D occupancy map (.bt).

OctoMap is the standard 3D occupancy format for robots that plan in 3D,
drones in particular: planners and `octomap_server` (ROS) load a .bt file
directly. This adapter voxelises the session's obstacles:

    session_YYYYMMDD_HHMMSS/
        scene.json
        models/*.stl
        octomap.bt         ← written here

Each voxel (default 5 cm) is one of:
  occupied — its cube overlaps an obstacle or an exported wall (tested
             against the exact hull, so obstacles are never thinner than
             they are), or lies just under the floor (floor=True);
  free     — inside the room boundary (floor to ceiling) and not occupied
             (free_space=True) — "known empty", as a sensor would report;
  unknown  — everything else.

Frame: the session's world frame (Z-up, metres, OptiTrack origin).
The file is written by the OctoMap library itself (octomap-python).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from ..collision import Box, CollisionWorld
from ..scene.scene_writer import load_session

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION_M = 0.05


def write_octomap(
    session:    Path | str,
    output:     Path | str | None = None,
    *,
    resolution: float = DEFAULT_RESOLUTION_M,
    free_space: bool = True,
    floor:      bool = True,
    walls:      bool = True,
) -> Path:
    """
    Write an OctoMap (.bt) for an exported session.

    Parameters
    ----------
    session    : session directory, or its scene.json
    output     : file to write; default <session>/octomap.bt
    resolution : voxel edge length in metres
    free_space : mark the room (inside the boundary) as free, not unknown
    floor      : mark one voxel layer under Z = 0 as occupied
    walls      : include exported boundary walls as occupied

    Returns the path written.
    """
    import octomap  # octomap-python

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    session = Path(session)
    loaded = load_session(session)
    output = Path(output) if output else loaded.session_dir / "octomap.bt"
    world = CollisionWorld.from_session(loaded, walls=walls)

    occupied = occupied_voxels(world, resolution)
    if floor:
        occupied = np.vstack([occupied, _floor_layer(world, resolution)])

    tree = octomap.OcTree(resolution)
    if free_space and world.has_bounds:
        free = _free_voxels(world, resolution)
        if len(free):
            tree.updateNodes(free, False)
    if len(occupied):
        tree.updateNodes(occupied, True)    # after free: occupied wins

    output.parent.mkdir(parents=True, exist_ok=True)
    tree.writeBinary(str(output).encode())  # prunes to max-likelihood
    logger.info(
        f"Wrote OctoMap: {output} ({len(occupied)} occupied voxels at "
        f"{resolution * 100:g} cm, {tree.size()} nodes)"
    )
    return output


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------

def _centres(lo, hi, res) -> np.ndarray:
    """
    Centres of the OctoMap voxels overlapping the box [lo, hi]. OctoMap
    voxels are aligned to multiples of `res` from the origin: voxel k spans
    [k*res, (k+1)*res).
    """
    k_lo = np.floor(np.asarray(lo) / res).astype(int)
    k_hi = np.ceil(np.asarray(hi) / res).astype(int)
    axes = [(np.arange(a, b) + 0.5) * res for a, b in zip(k_lo, k_hi, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def occupied_voxels(world: CollisionWorld, resolution: float) -> np.ndarray:
    """
    Centres (N, 3) of the voxels whose cube overlaps any solid in `world`.
    A voxel that only touches a solid's face (zero overlap, e.g. a wall face
    exactly on a voxel boundary) is not occupied — otherwise obstacles would
    grow by a voxel whenever a face lands on the grid. Only voxels near each
    solid are tested.
    """
    side = resolution * (1.0 - 1e-6)            # overlap, not mere contact
    voxel = Box((side, side, side))
    found = []
    for lo, hi in world.object_extents():
        for c in _centres(lo - resolution, hi + resolution, resolution):
            if world.collides(voxel, c):
                found.append(c)
    if not found:
        return np.empty((0, 3))
    keys = np.unique(np.floor(np.array(found) / resolution).astype(int), axis=0)   # dedupe
    return (keys + 0.5) * resolution


def _free_voxels(world: CollisionWorld, res: float) -> np.ndarray:
    """Centres of the voxels whose centre is inside the room boundary."""
    lo, hi = world.bounds()
    centres = _centres(lo, hi, res)
    return centres[world.in_bounds_many(centres)]


def _floor_layer(world: CollisionWorld, res: float) -> np.ndarray:
    """One voxel layer just under Z = 0, over the room (or the obstacles' extent)."""
    lo, hi = world.bounds()
    return _centres((lo[0], lo[1], -res), (hi[0], hi[1], 0.0), res)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="obstacle-builder-octomap",
        description="Write an OctoMap (octomap.bt) for an exported session.",
    )
    parser.add_argument("session", type=Path, help="session directory, or its scene.json")
    parser.add_argument("-o", "--output", type=Path,
                        help="file to write (default: <session>/octomap.bt)")
    parser.add_argument("-r", "--resolution", type=float, default=DEFAULT_RESOLUTION_M,
                        help=f"voxel size in metres (default {DEFAULT_RESOLUTION_M})")
    parser.add_argument("--no-free-space", action="store_true",
                        help="leave the room unknown instead of free")
    parser.add_argument("--no-floor", action="store_true",
                        help="do not mark the floor as occupied")
    parser.add_argument("--no-walls", action="store_true",
                        help="do not include exported boundary walls")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        path = write_octomap(
            args.session, args.output,
            resolution = args.resolution,
            free_space = not args.no_free_space,
            floor      = not args.no_floor,
            walls      = not args.no_walls,
        )
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
