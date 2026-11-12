# OptiTrack Obstacle Builder

[![CI](https://github.com/ayalon-mrslab/optitrack-obstacle-builder/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ayalon-mrslab/optitrack-obstacle-builder/actions/workflows/ci.yml)

Reads rigid bodies from an OptiTrack/Motive system via NatNet 4.x, reconstructs
3D geometry from marker positions, and exports STL models + a `scene.json` pose
file for use in Gazebo, ROS 2, Unity, or any other simulator.

## Requirements

- Python 3.10 or newer
- OptiTrack Motive 3.x with NatNet 4.x streaming enabled
- Host PC connected to the OptiTrack PC via Ethernet

## Setup

```bash
git clone https://github.com/ayalon-mrslab/optitrack-obstacle-builder
cd optitrack-obstacle-builder
python -m venv .venv

# Ubuntu / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e .
```

## Run

```bash
obstacle-builder
```

Or directly:

```bash
python scripts/run.py
```

- **File → Open session…** (Ctrl+O) — view a previously exported session:
  pick its `scene.json`. The obstacles and boundary are shown as saved; it is
  read-only (a saved session has no markers, so nothing can be recomputed or
  re-exported). *Disconnect* closes it.
- **Dev → Load mock scene** — tick to load built-in sample data (3 obstacles,
  4 cameras) and try the tool without OptiTrack; untick to remove it. It is
  not a saved session.

Connecting to OptiTrack, opening a session and the mock scene replace each
other — only one is shown at a time. The dot in the toolbar shows which:
green = live OptiTrack, amber = mock scene, blue = saved session.

## Motive setup

Before connecting, check these in Motive:

- **Publish only the relevant rigid bodies.** In the Assets pane, enable
  streaming only for the rigid bodies that are obstacles for this session
  (plus `boundary`, if you mark an interest area — see below). Every published rigid body with visible
  markers becomes an obstacle in the export.
- **Empty rigid bodies are ignored.** A rigid body that is published but has no
  visible markers (e.g. a drone's asset while the drone is not in the volume)
  is silently left out. The same applies to a real obstacle whose markers are
  all occluded at capture time, so if an obstacle is missing, check that
  Motive can see its markers.
- **Streaming:** Settings → Streaming → enable NatNet, transmission type
  **Multicast**.
- **Up Axis:** leave Settings → Streaming → Up Axis at **Z Axis** (the lab's
  configuration). The tool expects Z-up from Motive; if this setting is ever
  changed, obstacles appear rotated 90° relative to the floor (see
  `DEFAULT_STREAM_UP_AXIS` in `src/obstacle_builder/natnet/client.py`).

The rigid body names in Motive become the obstacle names and STL file names
(they can be renamed in the tool before export).

## Room boundary

The boundary is the floor of the area the obstacles live in, at Z = 0.
Choose how it is built with *Room boundary → Source*:

| Source | Floor shape | Use it when |
|---|---|---|
| **4 corner cameras** (default) | Rectangle through the 4 corner cameras — the 4 cameras enclosing the largest area, i.e. the physical corners even if the truss is rotated relative to Motive's axes. | The cameras sit on a rectangular truss, like the lab's. |
| **All cameras** | Polygon: the outline of all cameras (N sides; cameras inside the outline are ignored). | The room has no 4 real corners — circular rig, fewer cameras, irregular layout. |
| **Interest area** | Polygon: the outline of the markers of a rigid body named `boundary`, placed on the floor at the area's corners (4 markers → rectangle). | You only want part of the room. See `docs/marker_placement_guide.md`. |

Other boundary options:

- **Generate walls** — a thin wall panel on every floor edge, with the given
  **Height**, extruded **Outward** (toward the physical walls, default) or
  **Inward** (into the area). Changes apply immediately.
- **Show boundary in preview** — display only. The boundary (and walls, if
  enabled) is always included in the export.

All cameras are shown in the 3D preview as small dots; the ones at the
boundary's corners are highlighted. After changing rigid bodies in Motive
(e.g. adding the `boundary` body), press **Refresh**.

See `docs/scene_json_spec.md` for how the boundary is stored.

## Use in Gazebo

Turn an exported session into a Gazebo world:

```bash
obstacle-builder-gazebo ~/scenes/session_20261112_103000
gz sim ~/scenes/session_20261112_103000/world.sdf
```

Or tick **Also write Gazebo world** in the Export panel to get it with every
export. `world.sdf` is written into the session folder (mesh paths are relative, so
the folder can be moved or shared). Every obstacle is a static model whose
STL is used for both the look and the collisions; the boundary floor is
drawn semi-transparent (not solid) and exported walls are solid. A ground
plane and a light are added. No Gazebo plugins are set, so Gazebo uses its
defaults — to use the obstacles in your own simulation (e.g. with a robot),
copy the `<model>` blocks into your world or add your plugins to this one.

Options: `-o FILE` (write elsewhere), `--name` (world name),
`--no-ground-plane`, `--no-boundary`. Works with modern Gazebo (Fortress,
Garden, Harmonic, …). From Python: `obstacle_builder.adapters.gazebo.write_world()`.

## Use as an OctoMap (3D occupancy map)

OctoMap is the standard 3D map for robots that plan in 3D — drones in
particular. Turn a session into one:

```bash
obstacle-builder-octomap ~/scenes/session_20261112_103000      # → octomap.bt
```

Or tick **Also write OctoMap** in the Export panel. The map is written into
the session folder, in the session's frame (Z-up, metres, OptiTrack origin),
with 5 cm voxels (`-r` to change). Each voxel is:

| State | Where |
|---|---|
| occupied | overlaps an obstacle or an exported wall (partly covered voxels count, so obstacles are never thinner than they are), and one voxel layer under the floor |
| free | the rest of the room, inside the boundary from floor to ceiling |
| unknown | everything else |

Options: `--no-floor`, `--no-free-space` (leave the room unknown),
`--no-walls`, `-o FILE`. From Python: `obstacle_builder.adapters.octomap.write_octomap()`.

Serve it in ROS 2 with `octomap_server` (topics `octomap_binary`,
`octomap_full`, `occupied_cells_vis_array` for RViz, …):

```bash
ros2 run octomap_server octomap_server_node --ros-args \
    -p octomap_path:=$HOME/scenes/session_20261112_103000/octomap.bt \
    -p frame_id:=map
```

`frame_id` must name the TF frame that matches the OptiTrack origin in your
setup.

To view it in RViz, add the **OccupancyGrid** display from
`octomap_rviz_plugins` (`sudo apt install ros-<distro>-octomap-rviz-plugins`)
and set:

- Topic `/octomap_full`, Durability Policy **Transient Local**. The map is
  published once, so a Volatile subscriber never receives it.
- **Voxel Rendering**: Occupied, Free or All Voxels. Voxel Coloring Z-Axis
  makes the height easier to read.

Use OccupancyGrid, not OccupancyMap: OccupancyMap shows a flattened 2D
projection. The server's `free_cells_vis_array` markers are fully
transparent (octomap_server 2.3.1), so the free space shows only in the
OccupancyGrid display.

Planners that read `.bt` files (e.g. many drone planners) can load
the file directly. `obstacle-builder-ros2` (below) also publishes it, with
no `octomap_server` needed.

## Use in ROS 2

`obstacle-builder-ros2` publishes a session to ROS 2: the obstacles for
RViz, as MoveIt collision objects, and its OctoMap.

```bash
source /opt/ros/humble/setup.bash             # before running, in every terminal
obstacle-builder-ros2 ~/scenes/session_20261112_103000
obstacle-builder-ros2 ~/scenes/session_20261112_103000 --live   # follow Motive
```

| Topic | Type | What |
|---|---|---|
| `/obstacle_builder/markers` | `visualization_msgs/MarkerArray` | obstacle hulls with their names, boundary walls, floor outline |
| `/planning_scene` | `moveit_msgs/PlanningScene` (diff) | each obstacle and wall as a collision object (mesh), for MoveIt's `move_group` |
| `/obstacle_builder/octomap` | `octomap_msgs/Octomap` | the session's `octomap.bt`, if it was written |

Everything is in the frame given by `--frame-id` (default `map`), which
must match the OptiTrack origin in your TF tree.

- **Markers and OctoMap are latched** (transient local). In RViz, set
  the display's Durability Policy to Transient Local. View the OctoMap with
  the OccupancyGrid display, as described above.
- **MoveIt** gets the scene again whenever a new subscriber appears, so
  `move_group` can start before or after this node. The MoveIt output needs
  `moveit_msgs` (`sudo apt install ros-<distro>-moveit-msgs`); without it,
  only the MoveIt output is skipped.
- **`--live`** connects to Motive (`--server-ip`) and moves each obstacle to
  the live pose of the rigid body with the same name, at up to `--rate` Hz
  (default 10). Shapes stay as exported. The OctoMap still shows the saved
  poses.

Options: `--no-walls`, `--no-moveit`, `--no-octomap`. ROS arguments go
after `--ros-args`, for example `--ros-args -r __ns:=/lab`.

`rclpy` comes from ROS, not from pip. Create the virtual environment with
the Python of your ROS distribution (3.10 for Humble, 3.12 for Jazzy), and
source ROS before running.

## Use in a planner (3D collision checks)

```python
from obstacle_builder.collision import CollisionWorld, Sphere

world = CollisionWorld.from_session("session_20261112_103000", floor=True)
world.collides(Sphere(0.25), (1.0, 0.5, 1.2))          # is the pose free?
world.segment_free((0, 0, 1), (2, 1, 1), radius=0.25)  # is the straight move free?
```

The obstacles (and exported walls) are solid 3D objects for any robot shape
(sphere, box, cylinder, capsule): collision, clearance distance, exact
straight-line moves, room bounds. Works with OMPL or any other planner —
see `docs/collision.md` for the full guide, the safety-margin advice, a C++
(FCL) recipe, and `examples/ompl_planning.py`, a tested OMPL template for a
drone and a ground robot (`pip install -e ".[ompl]"`, Linux / macOS).

## Coordinate convention

| Property | Value |
|---|---|
| Up axis | Z |
| World origin | OptiTrack origin |
| Units | metres |
| Quaternion format | scalar-last `[qx, qy, qz, qw]` |

See `docs/scene_json_spec.md` for the full `scene.json` schema.

## Obstacle placement guideline

See `docs/marker_placement_guide.md` for how to place reflective markers
on physical obstacles so the tool can reconstruct their geometry.

## Output structure

Each export creates a session folder inside the directory you choose:

```text
<chosen directory>/
└── session_YYYYMMDD_HHMMSS/
    ├── scene.json              world poses + metadata for every model
    ├── models/
    │   ├── <obstacle>.stl      one per obstacle: convex hull, body-local frame
    │   ├── boundary.stl        floor polygon of the room boundary
    │   └── boundary_wall_N.stl wall panels (only with "Generate walls")
    ├── world.sdf               Gazebo world (only with "Also write Gazebo world")
    └── octomap.bt              3D occupancy map (only with "Also write OctoMap")
```

The STLs are centred on each body's origin; `scene.json` places them in the
world. See `docs/scene_json_spec.md`.

## Supported platforms

- Ubuntu 20.04+
- macOS 12+
- Windows 10+