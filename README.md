# OptiTrack Obstacle Builder

Reads rigid bodies from an OptiTrack/Motive system via NatNet 4.x, reconstructs
3D geometry from marker positions, and exports STL models + a `scene.json` pose
file for use in Gazebo, ROS 2, Unity, or any other simulator.

## Requirements

- Python 3.10 or newer
- OptiTrack Motive 3.x with NatNet 4.x streaming enabled
- Host PC connected to the OptiTrack PC via Ethernet

## Setup

```bash
git clone https://github.com/<your-lab>/optitrack-obstacle-builder
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

## Motive setup

Before connecting, check these in Motive:

- **Publish only the relevant rigid bodies.** In the Assets pane, enable
  streaming only for the rigid bodies that are obstacles for this session
  (plus `boundary`, if you use one). Every published rigid body with visible
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

## Coordinate convention

| Property | Value |
|---|---|
| Default up axis | Z (switchable to Y in GUI) |
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