# scene.json specification

Every export session produces a `scene.json` alongside the STL files.
This file is the universal interface between the tool and any downstream
consumer (Gazebo, ROS 2, Unity, custom scripts).

## Coordinate convention

| Property | Default | Optional |
|---|---|---|
| Up axis | Z | Y (selectable in GUI) |
| World origin | OptiTrack origin | — |
| Units | metres | — |
| Quaternion order | scalar-last `[qx, qy, qz, qw]` | — |
| Axis mapping | OptiTrack X→X, OptiTrack Z→Y, OptiTrack Y→Z | — |

The `convention` field in the file tells consumers which mode was used.

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
    "position": [0.0, 0.0, 0.0],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
    "room_dimensions": {
      "width_x": 6.0,
      "depth_y": 4.0,
      "height_z": 3.1
    },
    "wall_extrusion": null,
    "corner_camera_positions": [
      [-3.0, -2.0, 3.1],
      [ 3.0, -2.0, 3.1],
      [ 3.0,  2.0, 3.1],
      [-3.0,  2.0, 3.1]
    ]
  }
}
```

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

## Consuming scene.json in Gazebo (standalone)

The Gazebo adapter (see `src/obstacle_builder/adapters/`) reads `scene.json`
and generates a `.sdf` world file with one `<model>` per entry.