"""
Plan a collision-free path through an exported session with OMPL.

A template for using the obstacles in your own planner. OMPL does the
search; obstacle_builder.collision answers "is this pose / this motion
free?". Two robots:

  drone  — a sphere moving freely in 3D (state: x, y, z). Straight moves are
           checked exactly (swept capsule), not by sampling.
  ground — a box on the floor, like a quadruped or wheeled robot
           (state: x, y, heading).

Install OMPL first (Linux / macOS):  pip install -e ".[ompl]"

    python examples/ompl_planning.py SESSION --robot drone \\
        --start -3 -3 1 --goal 3 3 1.5 --radius 0.25
    python examples/ompl_planning.py SESSION --robot ground \\
        --start -3 -3 0 --goal 3 3 1.57 --size 0.7 0.31 0.4

SESSION is an exported session folder (or its scene.json). With
--output path.json the waypoints are saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
from scipy.spatial.transform import Rotation

from obstacle_builder.collision import Box, CollisionWorld, Sphere

# ---------------------------------------------------------------------------
# Drone: sphere in 3D
# ---------------------------------------------------------------------------

class _SegmentMotionValidator(ob.MotionValidator):
    """Exact straight-line check for a sphere, via CollisionWorld.segment_free."""

    def __init__(self, si, world: CollisionWorld, radius: float):
        super().__init__(si)
        self._world = world
        self._radius = radius

    def checkMotion(self, s1, s2, *_last_valid):
        return self._world.segment_free(
            (s1[0], s1[1], s1[2]), (s2[0], s2[1], s2[2]), self._radius
        )


def plan_drone(world: CollisionWorld, start, goal, radius: float,
               time_s: float = 5.0, margin: float = 0.05) -> list[tuple[float, float, float]] | None:
    """
    Waypoints (x, y, z) from start to goal for a sphere of `radius`, or None.
    The path keeps at least `margin` metres from every obstacle.
    """
    radius = radius + margin       # plan with the inflated robot
    space = ob.RealVectorStateSpace(3)
    lo, hi = world.bounds(margin=radius)
    bounds = ob.RealVectorBounds(3)
    for i in range(3):
        bounds.setLow(i, float(lo[i]))
        bounds.setHigh(i, float(hi[i]))
    space.setBounds(bounds)

    robot = Sphere(radius)
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    ss.setStateValidityChecker(
        lambda s: world.in_bounds((s[0], s[1], s[2]))
        and not world.collides(robot, (s[0], s[1], s[2]))
    )
    validator = _SegmentMotionValidator(si, world, radius)
    si.setMotionValidator(validator)

    path = _solve(ss, space, start, goal, time_s)
    return None if path is None else [(s[0], s[1], s[2]) for s in path]


# ---------------------------------------------------------------------------
# Ground robot: box on the floor, (x, y, heading)
# ---------------------------------------------------------------------------

def plan_ground(world: CollisionWorld, start, goal, size,
                time_s: float = 5.0, margin: float = 0.05) -> list[tuple[float, float, float]] | None:
    """
    Waypoints (x, y, heading) for a box robot of `size` = (length, width,
    height) standing on the floor, or None. The path keeps at least `margin`
    metres sideways from every obstacle. Motions are checked by sampling
    along them (OMPL's default) at ~1 cm; the margin also covers the tiny
    gaps between samples.
    """
    length, width, height = size
    length, width = length + 2 * margin, width + 2 * margin   # inflated footprint
    robot = Box((length, width, height))
    reach = math.hypot(length, width) / 2

    space = ob.SE2StateSpace()
    lo, hi = world.bounds(margin=reach)
    bounds = ob.RealVectorBounds(2)
    for i in range(2):
        bounds.setLow(i, float(lo[i]))
        bounds.setHigh(i, float(hi[i]))
    space.setBounds(bounds)

    def pose(s):
        x, y, yaw = s.getX(), s.getY(), s.getYaw()
        return (x, y, height / 2), Rotation.from_euler("z", yaw).as_quat()

    def valid(s):
        (x, y, z), q = pose(s)
        return world.in_bounds((x, y, 0.0)) and not world.collides(robot, (x, y, z), q)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(valid)
    si = ss.getSpaceInformation()
    extent = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1e-3)
    si.setStateValidityCheckingResolution(min(0.01 / extent, 0.05))

    def set_se2(state, xyyaw):
        state.setX(float(xyyaw[0]))
        state.setY(float(xyyaw[1]))
        state.setYaw(float(xyyaw[2]))

    path = _solve(ss, space, start, goal, time_s, setter=set_se2)
    return None if path is None else [(s.getX(), s.getY(), s.getYaw()) for s in path]


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _solve(ss, space, start, goal, time_s, setter=None):
    def set_vec(state, values):
        for i, v in enumerate(values):
            state[i] = float(v)

    setter = setter or set_vec
    s, g = space.allocState(), space.allocState()
    setter(s, start)
    setter(g, goal)
    si = ss.getSpaceInformation()
    for name, st in (("start", s), ("goal", g)):
        if not si.isValid(st):
            raise ValueError(f"The {name} pose is in collision or out of bounds.")
    ss.setStartAndGoalStates(s, g)
    ss.setPlanner(og.RRTConnect(si))
    if not ss.solve(time_s) or not ss.haveExactSolutionPath():
        return None
    ss.simplifySolution()
    path = ss.getSolutionPath()
    return [path.getState(i) for i in range(path.getStateCount())]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("--robot", choices=("drone", "ground"), default="drone")
    p.add_argument("--start", type=float, nargs=3, required=True,
                   metavar=("X", "Y", "Z_or_YAW"))
    p.add_argument("--goal", type=float, nargs=3, required=True,
                   metavar=("X", "Y", "Z_or_YAW"))
    p.add_argument("--radius", type=float, default=0.25, help="drone radius (m)")
    p.add_argument("--size", type=float, nargs=3, default=(0.7, 0.31, 0.4),
                   metavar=("L", "W", "H"), help="ground robot box (m)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="minimum clearance from obstacles (m)")
    p.add_argument("--time", type=float, default=5.0, help="planning time (s)")
    p.add_argument("--output", type=Path, help="save waypoints as JSON")
    args = p.parse_args(argv)

    ou.setLogLevel(ou.LogLevel.LOG_WARN)
    world = CollisionWorld.from_session(args.session, floor=args.robot == "drone")
    print(world)
    if args.robot == "drone":
        path = plan_drone(world, args.start, args.goal, args.radius, args.time, args.margin)
        labels = ("x", "y", "z")
    else:
        path = plan_ground(world, args.start, args.goal, args.size, args.time, args.margin)
        labels = ("x", "y", "yaw")
    if path is None:
        print("No path found.")
        return 1
    print(f"Path with {len(path)} waypoints ({', '.join(labels)}):")
    for wp in path:
        print("  " + "  ".join(f"{v:7.3f}" for v in wp))
    if args.output:
        args.output.write_text(json.dumps({"robot": args.robot, "fields": labels,
                                           "waypoints": path}, indent=2))
        print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
