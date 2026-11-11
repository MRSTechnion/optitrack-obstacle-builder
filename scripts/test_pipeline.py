"""
Headless pipeline test — no GUI, no hardware.
Run from repo root: python scripts/test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.scene.scene_writer import build_session, export_session

FIXTURE = Path("tests/fixtures/sample_scene.json")
OUTPUT  = Path("/tmp/obstacle_builder_test")

print("Loading fixture...")
client = MockNatNetConnection(str(FIXTURE))
client.connect()
frame = client.capture_frame()
print(f"  Frame #{frame.frame_number} — {frame.rigid_body_count} bodies, {len(frame.cameras)} cameras")

print("Converting to Z-up world frame...")
world = convert_frame(frame, z_up=True)

print("Building hulls...")
bodies = [b for b in world.rigid_bodies if b.name.lower() != "boundary"]
hulls  = build_all_hulls(bodies)
for name, hull in hulls.items():
    print(f"  {name}: {len(hull.vertices)} vertices, volume={hull.volume*1e6:.1f} cm³, dims={hull.dimensions.round(3)} m")

print("Building boundary...")
boundary = build_boundary(world.cameras, wall_height_m=3.0, wall_extrude_in=False)
if boundary:
    print(f"  Room: {boundary.room_width_x:.2f} m × {boundary.room_depth_y:.2f} m × {boundary.room_height_z:.2f} m")
    print(f"  Walls: {len(boundary.wall_meshes)} panels")

print("Exporting session...")
OUTPUT.mkdir(parents=True, exist_ok=True)
session     = build_session(hulls, boundary, world.rigid_bodies)
session_dir = export_session(session, OUTPUT)
print(f"  Exported to: {session_dir}")

print("\nAll done. Check output at:", session_dir)