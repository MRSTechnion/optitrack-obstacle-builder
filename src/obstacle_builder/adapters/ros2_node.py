"""
The ROS 2 node behind obstacle-builder-ros2 (see ros2.py). Needs a sourced
ROS 2 environment (rclpy, visualization_msgs); moveit_msgs and octomap_msgs
are optional.
"""

from __future__ import annotations

import logging
from array import array

import rclpy
from geometry_msgs.msg import Point, Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ..geometry.transform import convert_frame
from .ros2 import LivePoseTracker, SceneContent, SceneObject, read_octomap_file

logger = logging.getLogger(__name__)

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)

try:
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from shape_msgs.msg import Mesh, MeshTriangle
    HAVE_MOVEIT = True
except ImportError:
    HAVE_MOVEIT = False

try:
    from octomap_msgs.msg import Octomap
    HAVE_OCTOMAP = True
except ImportError:
    HAVE_OCTOMAP = False


def _pose(position, orientation) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = (float(v) for v in orientation)
    return p


def _points(xyz) -> list[Point]:
    return [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz]


class ObstacleNode(Node):
    """
    Publishes a SceneContent; with a NatNet `client`, follows live poses.
    `client` is anything with a `latest_frame` property (raw NatNet frames).
    """

    def __init__(self, content: SceneContent, frame_id: str = "map", *,
                 moveit: bool = True, client=None, rate_hz: float = 10.0):
        super().__init__("obstacle_builder")
        self._content = content
        self._frame_id = frame_id
        self._client = client

        self._markers_pub = self.create_publisher(MarkerArray, "~/markers", LATCHED)

        self._scene_pub = None
        if moveit and HAVE_MOVEIT:
            self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
            self._scene_subs = 0
            self.create_timer(0.5, self._resend_scene_to_new_subscribers)
        elif moveit:
            self.get_logger().warning("moveit_msgs not installed: not publishing to MoveIt "
                                      "(sudo apt install ros-$ROS_DISTRO-moveit-msgs)")

        self._octomap_pub = None
        if content.octomap_path is not None:
            if HAVE_OCTOMAP:
                self._octomap_pub = self.create_publisher(Octomap, "~/octomap", LATCHED)
                self._octomap_pub.publish(self._octomap_msg())
            else:
                self.get_logger().warning("octomap_msgs not installed: not publishing the OctoMap")

        self._tracker = None
        if client is not None:
            self._tracker = LivePoseTracker(
                {o.name: (o.position, o.orientation) for o in content.obstacles})
            self.create_timer(1.0 / rate_hz, self._follow_live_poses)
            if self._octomap_pub is not None:
                self.get_logger().info("The OctoMap shows the saved poses; it is not updated live.")

        self._markers_pub.publish(self.markers())
        self.get_logger().info(
            f"Publishing {len(content.obstacles)} obstacles, {len(content.walls)} walls"
            f"{', OctoMap' if self._octomap_pub else ''}"
            f"{', MoveIt scene' if self._scene_pub else ''} in frame '{frame_id}'"
            f"{' (live)' if client is not None else ''}"
        )

    # -- messages ---------------------------------------------------------

    def _header_marker(self, ns: str, mid: int, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        return m

    def markers(self) -> MarkerArray:
        arr = MarkerArray()
        for i, o in enumerate(self._content.obstacles):
            m = self._header_marker("obstacles", i, Marker.TRIANGLE_LIST)
            m.pose = _pose(o.position, o.orientation)
            m.points = _points(o.vertices[o.faces].reshape(-1, 3))
            m.color = ColorRGBA(r=o.color[0], g=o.color[1], b=o.color[2], a=o.color[3])
            arr.markers.append(m)

            label = self._header_marker("names", i, Marker.TEXT_VIEW_FACING)
            label.pose = _pose(o.position, (0, 0, 0, 1))
            label.pose.position.z += float(o.vertices[:, 2].max()) + 0.15
            label.scale.z = 0.12
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = o.name
            arr.markers.append(label)

        for i, w in enumerate(self._content.walls):
            m = self._header_marker("walls", i, Marker.TRIANGLE_LIST)
            m.points = _points(w.vertices[w.faces].reshape(-1, 3))
            m.color = ColorRGBA(r=w.color[0], g=w.color[1], b=w.color[2], a=w.color[3])
            arr.markers.append(m)

        floor = self._content.floor_polygon
        if floor is not None and len(floor):
            m = self._header_marker("boundary", 0, Marker.LINE_STRIP)
            m.scale.x = 0.02
            m.points = _points(list(floor) + [floor[0]])
            m.color = ColorRGBA(r=0.3, g=0.65, b=1.0, a=1.0)
            arr.markers.append(m)
        return arr

    def _collision_object(self, o: SceneObject):
        co = CollisionObject()
        co.header.frame_id = self._frame_id
        co.id = o.name
        co.pose = _pose(o.position, o.orientation)
        mesh = Mesh()
        mesh.vertices = _points(o.vertices)
        mesh.triangles = [MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in o.faces]
        co.meshes = [mesh]
        co.mesh_poses = [_pose((0, 0, 0), (0, 0, 0, 1))]
        co.operation = CollisionObject.ADD        # ADD replaces an existing object
        return co

    def planning_scene(self, objects: list[SceneObject] | None = None):
        """A PlanningScene diff adding `objects` (default: all obstacles and walls)."""
        if objects is None:
            objects = self._content.obstacles + self._content.walls
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [self._collision_object(o) for o in objects]
        return scene

    def _octomap_msg(self):
        tree_id, res, data = read_octomap_file(self._content.octomap_path)
        msg = Octomap()
        msg.header.frame_id = self._frame_id
        msg.binary, msg.id, msg.resolution = True, tree_id, res
        msg.data = array("b", data)
        return msg

    # -- timers -----------------------------------------------------------

    def _resend_scene_to_new_subscribers(self) -> None:
        n = self._scene_pub.get_subscription_count()
        if n > self._scene_subs:          # e.g. move_group (re)started
            self._scene_pub.publish(self.planning_scene())
        self._scene_subs = n

    def _follow_live_poses(self) -> None:
        raw = self._client.latest_frame
        if raw is None:
            return
        moved = self._tracker.update(convert_frame(raw))
        if not moved:
            return
        for o in self._content.obstacles:
            if o.name in moved:
                o.position, o.orientation = moved[o.name]
        self._markers_pub.publish(self.markers())
        if self._scene_pub is not None:
            self._scene_pub.publish(self.planning_scene(
                [o for o in self._content.obstacles if o.name in moved]))


def run(content: SceneContent, *, frame_id: str, live: bool, server_ip: str,
        rate_hz: float, moveit: bool, ros_args: list[str]) -> int:
    client = None
    if live:
        from ..natnet.client import NatNetConnection
        client = NatNetConnection(server_ip=server_ip)
        if not client.connect():
            logger.error("Could not connect to Motive; see the log above.")
            return 1

    rclpy.init(args=["obstacle-builder-ros2", *ros_args])
    node = ObstacleNode(content, frame_id, moveit=moveit, client=client, rate_hz=rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if client is not None:
            client.disconnect()
    return 0
