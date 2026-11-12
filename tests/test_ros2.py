"""
Tests for the ROS 2 adapter (adapters/ros2.py, adapters/ros2_node.py).

The ROS-free parts always run. The node tests need a sourced ROS 2
environment (rclpy) and are skipped otherwise; the MoveIt test also needs
moveit_msgs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from obstacle_builder.adapters import octomap as octomap_adapter
from obstacle_builder.adapters import ros2
from obstacle_builder.geometry.hull import build_all_hulls, build_boundary
from obstacle_builder.geometry.transform import convert_frame
from obstacle_builder.natnet.client import MockNatNetConnection
from obstacle_builder.natnet.types import Frame, Quaternion, RigidBody, Vector3
from obstacle_builder.scene.scene_writer import build_session, export_session, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_scene.json"

try:
    import rclpy  # noqa: F401
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False

needs_ros = pytest.mark.skipif(not HAVE_ROS, reason="ROS 2 (rclpy) not sourced")


@pytest.fixture
def session(tmp_path):
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    w = convert_frame(client.capture_frame())
    s = build_session(build_all_hulls(w.rigid_bodies),
                      build_boundary(w.cameras, wall_height_m=1.0), w.rigid_bodies, wall_height=1.0)
    return export_session(s, tmp_path)


# ---------------------------------------------------------------------------
# ROS-free parts
# ---------------------------------------------------------------------------

def test_scene_content(session):
    c = ros2.scene_content(session)
    loaded = load_session(session)
    assert [o.name for o in c.obstacles] == [b.name for b in loaded.bodies]
    assert len(c.walls) == len(loaded.boundary.wall_meshes) > 0
    assert all(w.is_wall for w in c.walls)
    assert c.floor_polygon.shape[1] == 3
    assert c.octomap_path is None                       # not written yet
    o = c.obstacles[0]
    assert o.faces.max() < len(o.vertices)


def test_scene_content_options(session):
    octomap_adapter.write_octomap(session, resolution=0.1)
    assert ros2.scene_content(session).octomap_path == session / "octomap.bt"
    c = ros2.scene_content(session, walls=False, octomap=False)
    assert c.walls == [] and c.octomap_path is None


def test_read_octomap_file(session, tmp_path):
    path = octomap_adapter.write_octomap(session, resolution=0.1)
    tree_id, res, data = ros2.read_octomap_file(path)
    assert tree_id == "OcTree" and res == pytest.approx(0.1)
    assert path.read_bytes().endswith(data) and len(data) > 0

    bad = tmp_path / "bad.bt"
    bad.write_bytes(b"not an octomap")
    with pytest.raises(ValueError):
        ros2.read_octomap_file(bad)


def _frame(name, pos, quat=(0, 0, 0, 1), valid=True):
    body = RigidBody(body_id=1, name=name, position=Vector3(*pos), orientation=Quaternion(*quat),
                     markers=[], tracking_valid=valid, mean_error=0.0)
    return Frame(frame_number=1, timestamp=0.0, rigid_bodies=[body], cameras=[])


def test_live_pose_tracker():
    t = ros2.LivePoseTracker({"box": ((0, 0, 0), (0, 0, 0, 1))})
    assert t.update(_frame("box", (0.0005, 0, 0))) == {}                     # below tolerance
    assert t.update(_frame("other", (5, 0, 0))) == {}                        # not an obstacle
    assert t.update(_frame("box", (1, 0, 0), valid=False)) == {}             # not tracked
    moved = t.update(_frame("box", (0.1, 0, 0)))
    assert moved["box"][0] == pytest.approx((0.1, 0, 0))
    yawed = tuple(Rotation.from_euler("z", 1, degrees=True).as_quat())
    assert "box" in t.update(_frame("box", (0.1, 0, 0), yawed))              # rotation only
    assert t.update(_frame("box", (0.1, 0, 0), yawed)) == {}                 # unchanged


@pytest.mark.skipif(HAVE_ROS, reason="checks the message shown without ROS")
def test_cli_without_ros(session, capsys):
    assert ros2.main([str(session)]) == 1
    assert "Source your ROS 2 environment" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The node (needs ROS 2)
# ---------------------------------------------------------------------------

@pytest.fixture
def ros():
    import rclpy
    rclpy.init()
    yield rclpy
    rclpy.shutdown()


def _receive(rclpy, node, msg_type, topic, qos, timeout=5.0, until=None):
    """Spin `node` and a listener until a message on `topic` satisfies `until` (default: any)."""
    got = []
    listener = rclpy.create_node("test_listener")
    listener.create_subscription(msg_type, topic, got.append, qos)
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(listener)
    deadline = time.monotonic() + timeout
    until = until or (lambda _msg: True)
    while not any(until(m) for m in got) and time.monotonic() < deadline:
        ex.spin_once(timeout_sec=0.05)
    listener.destroy_node()
    return got


@needs_ros
def test_markers_latched(ros, session):
    from visualization_msgs.msg import Marker, MarkerArray

    from obstacle_builder.adapters.ros2_node import LATCHED, ObstacleNode
    content = ros2.scene_content(session)
    node = ObstacleNode(content, "world", moveit=False)
    msgs = _receive(ros, node, MarkerArray, "/obstacle_builder/markers", LATCHED)
    node.destroy_node()
    assert msgs, "no MarkerArray received"
    by_ns = {}
    for m in msgs[0].markers:
        by_ns.setdefault(m.ns, []).append(m)
        assert m.header.frame_id == "world"
    assert len(by_ns["obstacles"]) == len(by_ns["names"]) == len(content.obstacles)
    assert len(by_ns["walls"]) == len(content.walls)
    assert by_ns["boundary"][0].type == Marker.LINE_STRIP
    o, m = content.obstacles[0], by_ns["obstacles"][0]
    assert len(m.points) == 3 * len(o.faces)
    assert (m.pose.position.x, m.pose.position.y, m.pose.position.z) == pytest.approx(o.position)
    assert {n.text for n in by_ns["names"]} == {o.name for o in content.obstacles}


@needs_ros
def test_octomap_topic(ros, session):
    pytest.importorskip("octomap_msgs")
    import octomap
    from octomap_msgs.msg import Octomap

    from obstacle_builder.adapters.ros2_node import LATCHED, ObstacleNode
    octomap_adapter.write_octomap(session, resolution=0.1)
    node = ObstacleNode(ros2.scene_content(session), "map", moveit=False)
    msgs = _receive(ros, node, Octomap, "/obstacle_builder/octomap", LATCHED)
    node.destroy_node()
    assert msgs, "no Octomap received"
    msg = msgs[0]
    assert msg.binary and msg.id == "OcTree" and msg.resolution == pytest.approx(0.1)
    # the message data is a complete tree: the file's body, readable by OctoMap
    data = bytes(np.asarray(msg.data, dtype=np.int8).tobytes())
    assert (session / "octomap.bt").read_bytes().endswith(data)
    tree = octomap.OcTree(0.1)
    assert tree.readBinary(str(session / "octomap.bt").encode())
    assert tree.size() > 0


@needs_ros
def test_moveit_planning_scene(ros, session):
    pytest.importorskip("moveit_msgs")
    from moveit_msgs.msg import CollisionObject, PlanningScene

    from obstacle_builder.adapters.ros2_node import ObstacleNode
    content = ros2.scene_content(session)
    node = ObstacleNode(content, "map")
    msgs = _receive(ros, node, PlanningScene, "/planning_scene", 10)   # resent to new subscribers
    node.destroy_node()
    assert msgs, "no PlanningScene received"
    scene = msgs[0]
    assert scene.is_diff
    ids = [co.id for co in scene.world.collision_objects]
    assert ids == [o.name for o in content.obstacles] + [w.name for w in content.walls]
    co, o = scene.world.collision_objects[0], content.obstacles[0]
    assert co.operation == CollisionObject.ADD and co.header.frame_id == "map"
    assert len(co.meshes[0].vertices) == len(o.vertices)
    assert len(co.meshes[0].triangles) == len(o.faces)
    assert co.pose.position.x == pytest.approx(o.position[0])


@needs_ros
def test_live_updates_follow_moved_body(ros, session):
    from visualization_msgs.msg import MarkerArray

    from obstacle_builder.adapters.ros2_node import LATCHED, ObstacleNode
    client = MockNatNetConnection(str(FIXTURE))
    client.connect()
    content = ros2.scene_content(session)
    node = ObstacleNode(content, "map", moveit=False, client=client, rate_hz=50)

    moved = client.latest_frame.rigid_bodies[0]
    moved.position = Vector3(moved.position.x + 0.2, moved.position.y, moved.position.z)
    expected = convert_frame(client.latest_frame).get_body_by_name(moved.name).position.as_tuple()

    i = [o.name for o in content.obstacles].index(moved.name)

    def at_live_pose(arr):
        m = next(m for m in arr.markers if m.ns == "obstacles" and m.id == i)
        return np.allclose((m.pose.position.x, m.pose.position.y, m.pose.position.z), expected)

    msgs = _receive(ros, node, MarkerArray, "/obstacle_builder/markers", LATCHED, until=at_live_pose)
    node.destroy_node()
    assert any(at_live_pose(m) for m in msgs), "no live update published"
