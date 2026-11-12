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
<chosen_directory>/
session_YYYYMMDD_HHMMSS/
models/

<obstacle_name>.stl ← geometry centred at origin, body-local frame
boundary.stl ← capture volume floor polygon
scene.json ← world poses for every model


## Supported platforms

- Ubuntu 20.04+
- macOS 12+
- Windows 10+