# scene.json specification

Every export session produces a `scene.json` alongside the STL files.
This file is the universal interface between the tool and any downstream
consumer (Gazebo, ROS 2, Unity, custom scripts).

## Coordinate convention

| Property | Value |
|---|---|
| Up axis | Z |
| World origin | OptiTrack origin |
| Units | metres |
| Quaternion order | scalar-last `[qx, qy, qz, qw]` |

This is the frame Motive uses with *Streaming → Up Axis = Z*. All positions,
orientations and meshes in a session use it. The
`convention` field is always `"z_up"`; readers reject anything else.

## Schema

```json
{
  "version": "1.0",
  "convention": "z_up",
  "origin": "optitrack",
  "units": "metres",
  "created": "2024-11-10T14:32:00",
  "models": [
    {
      "name": "wall_left",
      "stl": "models/wall_left.stl",
      "position": [1.2, 0.5, 0.0],
      "quaternion": [0.0, 0.0, 0.0, 1.0],
      "type": "obstacle",
      "marker_count": 8
    }
  ],
  "boundary": {
    "stl": "models/boundary.stl",
    "type": "boundary",
    "mode": "corners",
    "position": [0.0, 0.0, 0.0],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
    "room_dimensions": {
      "width_x": 6.0,
      "depth_y": 4.0,
      "height_z": 3.1
    },
    "wall_extrusion": null,
    "floor_polygon": [
      [-3.0, -2.0, 0.0],
      [ 3.0, -2.0, 0.0],
      [ 3.0,  2.0, 0.0],
      [-3.0,  2.0, 0.0]
    ],
    "corner_camera_positions": [
      [-3.0, -2.0, 3.1],
      [ 3.0, -2.0, 3.1],
      [ 3.0,  2.0, 3.1],
      [-3.0,  2.0, 3.1]
    ]
  }
}
```

## The `boundary` object

The floor of the capture volume, built from the camera positions projected
onto Z = 0. `boundary` is `null` if there were too few cameras.

| Field | Meaning |
|---|---|
| `mode` | How the floor was built: `corners`, `all_cameras` or `rigid_body` — see below. |
| `stl` | Floor polygon mesh (body-local = world, so the pose is identity). |
| `floor_polygon` | The floor polygon's vertices, counter-clockwise seen from above (+Z), at Z = 0. 4 entries in `corners` mode, N in `all_cameras` / `rigid_body` mode. |
| `room_dimensions.width_x`, `depth_y` | `corners`: the rectangle's side lengths (correct even if the truss is rotated relative to the OptiTrack axes). `all_cameras` / `rigid_body`: the polygon's extent along X and Y. |
| `room_dimensions.height_z` | Mean height of the cameras at the polygon's vertices (≈ ceiling). In `rigid_body` mode: mean height of all cameras (0 if none). |
| `wall_extrusion` | `null`, or the optional wall panels: one STL per polygon edge, `thickness_m` thick, extruded `outward` (toward the physical walls) or `inward`. |
| `corner_camera_positions` | The cameras at the polygon's vertices (world frame, at their real height). Empty in `rigid_body` mode. |

### Modes

- **`corners`** (default) — for a rectangular truss. The tool picks the 4
  cameras that enclose the largest floor area (the physical corners, in any
  orientation) and the floor is that quadrilateral.
- **`all_cameras`** — for any other layout: a circular rig, fewer cameras,
  an irregular room. The floor is the outline (convex hull) of all cameras,
  an N-sided polygon; cameras inside the outline are ignored. No assumption
  is made about the room's shape.
- **`rigid_body`** — an interest area: the outline of the markers of a rigid
  body named `boundary`, placed on the floor at the area's corners (see
  `marker_placement_guide.md`). 4 markers give a quadrilateral, more give a
  polygon. Not tied to the cameras.

Selected in the GUI under *Room boundary → Source*: *4 corner cameras*
(`corners`), *All cameras* (`all_cameras`), *Interest area* (`rigid_body`).

## Loading a session in Python

`obstacle_builder.scene.scene_writer.load_session(path)` (session directory or
its `scene.json`) returns the hulls (vertices + faces from the STLs, body-local
frame), each model's world pose, and the boundary (floor polygon, walls,
corner cameras) — the same objects the tool works with. The GUI's
*File → Open session* uses it; adapters can too.

## Consuming scene.json in ROS 2

```python
import json
from geometry_msgs.msg import Pose

with open("scene.json") as f:
    scene = json.load(f)

for model in scene["models"]:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = model["position"]
    qx, qy, qz, qw = model["quaternion"]
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    # spawn model using pose + model["stl"]
```

## Consuming scene.json in Gazebo

`obstacle-builder-gazebo <session>` (or `adapters.gazebo.write_world()`)
writes `world.sdf` next to `scene.json`, with mesh URIs relative to it — the
session folder stays self-contained. `gz sim <session>/world.sdf` runs it.

| scene.json | world.sdf |
|---|---|
| each `models[]` entry | a static `<model>` named after it (sanitised to letters, digits, `_`, `-`; made unique), its STL as both `<visual>` and `<collision>` mesh, `<pose>` = `position` + `quaternion` as fixed-axis roll-pitch-yaw |
| `boundary.stl` | `boundary_floor`: visual only (semi-transparent, 1 mm above the ground) — the floor is not an obstacle |
| `boundary.wall_extrusion.stl_files` | `boundary_wall_N`: solid static models |

Also added: a ground plane at Z = 0 and a directional light. The world has no
`<plugin>` elements, so Gazebo loads its default systems; it is SDF 1.8, for
modern Gazebo (Fortress / Garden / Harmonic and later). Gazebo Classic is not
targeted.