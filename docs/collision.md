# Using the obstacles in a planner (collision layer)

An exported session (`scene.json` + STLs) describes solid 3D obstacles. Any
planner — for a drone, a ground robot or an arm — needs two questions
answered about them:

1. **Is this pose free?** (the robot, placed here, does not touch an obstacle)
2. **Is this motion free?** (the robot can move from A to B without touching one)

`obstacle_builder.collision` answers both, for every session, in Python. It
uses [FCL](https://github.com/flexible-collision-library/fcl), the collision
library behind MoveIt and most OMPL setups.

> **OMPL is not the collision checker.** OMPL is a library of planning
> algorithms (RRT, PRM, …). It searches for a path by repeatedly asking *your*
> code "is this state valid?" and "is this motion valid?". The collision layer
> is what answers those questions; OMPL is one of its users. Your own sampler,
> a custom A\*, or a safety check on a controller can use it just the same.

## Quick start

```python
from obstacle_builder.collision import CollisionWorld, Sphere, Box

world = CollisionWorld.from_session("session_20261112_103000", floor=True)

drone = Sphere(radius=0.25)
world.collides(drone, (1.0, 0.5, 1.2))            # True / False
world.distance(drone, (1.0, 0.5, 1.2))            # clearance in m (< 0: penetrating)
world.nearest(drone, (1.0, 0.5, 1.2))             # ("Box_01", 0.83)
world.segment_free((0, 0, 1), (2, 1, 1), 0.25)    # straight move: exact, not sampled

go2 = Box((0.7, 0.31, 0.4))                       # length, width, height
world.collides(go2, (1.0, 0.5, 0.2), (0, 0, 0.383, 0.924))   # yawed 45°

world.in_bounds((1.0, 0.5, 1.2))                  # inside the room boundary?
lo, hi = world.bounds(margin=0.25)                # box to sample in
```

All positions are in the session's world frame: **Z-up, metres**, origin =
OptiTrack origin. Orientations are quaternions `[qx, qy, qz, qw]` (the same
convention as `scene.json`).

## What is solid

| Object | Solid? | Option |
|---|---|---|
| Every obstacle (its convex hull, exactly as exported) | yes | — |
| Boundary walls (if exported with *Generate walls*) | yes | `walls=True` (default) |
| The floor, as everything below Z = 0 | optional | `floor=True` — use for drones; leave off for ground robots, which touch the floor |
| The boundary floor polygon | no — it is the *planning area* | use `in_bounds()` / `bounds()` |

## Robot shapes

| Shape | Parameters | Placed at |
|---|---|---|
| `Sphere(radius)` | | centre |
| `Box((x, y, z))` | full side lengths | centre |
| `Cylinder(radius, length)` | along local Z | centre |
| `Capsule(radius, length)` | along local Z; `length` = cylinder part only, caps add `radius` at each end | centre |

A robot that is not one of these: use a slightly larger enclosing shape, or
several calls (one per part). Advanced: `world.add_object(name, fcl_geometry,
position, orientation)` accepts any FCL geometry.

## Keep a safety margin

A planner that only asks "does it touch?" returns paths that graze obstacles
with millimetres to spare. Real robots need clearance — tracking errors, and
the hulls pass through the **marker centres**, so they are slightly smaller
than the real objects. Inflate the robot by a margin when planning (e.g.
`Sphere(radius + 0.05)`), or require `world.distance(...) >= margin`.

## Example: planning with OMPL

[`examples/ompl_planning.py`](../examples/ompl_planning.py) is a complete,
tested template for two robots:

- **drone** — a sphere in 3D. State validity = `in_bounds` and not `collides`;
  motion validity = `segment_free` (exact).
- **ground** — a box on the floor with a heading (OMPL's SE2 space), like the
  go2. Motions are checked by sampling at ~1 cm.

Both keep a 5 cm margin by default (`--margin`).

```bash
pip install -e ".[ompl]"      # Linux / macOS (OMPL has no Windows wheels)
python examples/ompl_planning.py SESSION --robot drone  --start -3 -3 1 --goal 3 3 1.5 --radius 0.25
python examples/ompl_planning.py SESSION --robot ground --start -3 -3 0 --goal 3 3 1.57 --size 0.7 0.31 0.4
```

## C++ recipe (FCL directly)

There is no C++ library in this project (yet); the session files map
directly onto C++ FCL:

1. Parse `scene.json` (e.g. with nlohmann/json). For each entry in `models`:
   read `models/<name>.stl` (body-local frame) and the world pose
   (`position`, `quaternion` = x, y, z, w).
2. Build the geometry. Each hull is convex, so `fcl::Convex<double>` is exact
   and fast; `fcl::BVHModel<fcl::OBBRSSd>` also works for any mesh.
3. Give it the pose and register it:

```cpp
#include <fcl/fcl.h>

// vertices: std::vector<fcl::Vector3d>; faces: flat [3, i, j, k, 3, ...]
auto verts = std::make_shared<std::vector<fcl::Vector3d>>(stl_vertices);
auto faces = std::make_shared<std::vector<int>>(stl_faces_flat);
auto hull  = std::make_shared<fcl::Convexd>(verts, n_triangles, faces);

fcl::Transform3d pose = fcl::Transform3d::Identity();
pose.translation() = fcl::Vector3d(px, py, pz);
pose.linear() = fcl::Quaterniond(qw, qx, qy, qz).toRotationMatrix();  // Eigen: w first!

auto obstacle = std::make_shared<fcl::CollisionObjectd>(hull, pose);

// query: a 0.25 m sphere robot at (x, y, z)
fcl::CollisionObjectd robot(std::make_shared<fcl::Sphered>(0.25),
                            fcl::Transform3d(Eigen::Translation3d(x, y, z)));
fcl::CollisionRequestd req;  fcl::CollisionResultd res;
bool hit = fcl::collide(&robot, obstacle.get(), req, res) > 0;
```

Note the quaternion order: `scene.json` is `[x, y, z, w]`; Eigen's
constructor takes `(w, x, y, z)`. For many obstacles, register them in an
`fcl::DynamicAABBTreeCollisionManagerd`. In **MoveIt**, each model becomes a
`moveit_msgs/CollisionObject` with a mesh (the STL) and the pose — that is
what the planned ROS 2 adapter will publish.
