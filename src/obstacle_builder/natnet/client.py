"""
NatNet 4.1 client wrapper.

Wraps the OptiTrack NatNetClient SDK into a clean three-method interface:

    client = NatNetConnection(server_ip="192.168.1.100")
    client.connect()
    frame = client.capture_frame()
    client.disconnect()

Required files in this same directory (copy from the OptiTrack PC):
    NatNetClient.py       — SDK core
    DataDescriptions.py   — description data structures
    MoCapData.py          — frame data structures

These are found at:
    C:\\Users\\User\\Downloads\\NatNet_SDK_4.1\\NatNetSDK\\Samples\\PythonClient\\
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..geometry.transform import zup_stream_to_native
from .types import CameraInfo, Frame, Marker, Quaternion, RigidBody, Vector3

# The NatNet SDK files in this folder (NatNetClient.py, MoCapData.py,
# DataDescriptions.py, PythonSample.py) are copied unmodified from the OptiTrack
# PC and must not be edited. NatNetClient.py does top-level
# `import DataDescriptions` / `import MoCapData`, so this folder has to be on
# sys.path. It is *appended* (not prepended) so that our own natnet/types.py
# never shadows the standard-library `types` module.
_NATNET_DIR = str(Path(__file__).resolve().parent)
if _NATNET_DIR not in sys.path:
    sys.path.append(_NATNET_DIR)

try:
    import DataDescriptions as _SDKDataDescriptions  # same module the SDK uses

    from .NatNetClient import NatNetClient as _NatNetSDK
    _SDK_AVAILABLE = True
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _e:
    _SDK_AVAILABLE = False
    _SDK_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)
_sdk_logger = logging.getLogger(f"{__name__}.sdk")


class _SDKPrintToLogging:
    """
    Replacement for print() inside the vendored NatNet SDK module.

    The SDK print()s unconditionally - e.g. "MoCap Frame: N" plus a blank
    line for every frame (~180/s) - regardless of print_level. Python looks
    up `print` in a module's globals before builtins, so we install this as
    `print` in the SDK module's namespace (the SDK file itself is untouched).

    - Routine output goes to DEBUG (hidden unless debug logging is enabled).
    - Lines starting with ERROR/WARNING (occasional frame decode failures)
      go to WARNING, rate-limited so they cannot flood the log.
    """

    def __init__(self, interval_s: float = 30.0) -> None:
        self._interval_s = interval_s
        self._last_warn  = float("-inf")
        self._suppressed = 0
        self._lock       = threading.Lock()

    def __call__(self, *args, sep=" ", end="\n", file=None, flush=False) -> None:
        msg = sep.join(str(a) for a in args).strip()
        if not msg:
            return
        if not msg.startswith(("ERROR", "WARNING")):
            _sdk_logger.debug(msg)
            return
        with self._lock:
            now = time.monotonic()
            if now - self._last_warn < self._interval_s:
                self._suppressed += 1
                _sdk_logger.debug(msg)
                return
            suppressed, self._suppressed = self._suppressed, 0
            self._last_warn = now
        extra = f" ({suppressed} similar SDK messages suppressed)" if suppressed else ""
        _sdk_logger.warning(f"NatNet SDK: {msg}{extra}")


if _SDK_AVAILABLE:
    sys.modules[_NatNetSDK.__module__].print = _SDKPrintToLogging()

# ---------------------------------------------------------------------------
# Network defaults — adjust server_ip to match your OptiTrack PC
# ---------------------------------------------------------------------------
DEFAULT_SERVER_IP     = "192.168.1.100"
# None = auto-detect the local interface that routes to the server (see
# _local_ip_for). Must be a real NIC address for multicast: joining the
# group on 127.0.0.1 means no frames are ever received.
DEFAULT_CLIENT_IP: str | None = None
DEFAULT_USE_MULTICAST = True
# Motive: Settings → Streaming → "Up Axis". The lab's Motive is configured
# with Z-up and this is not meant to be changed. It applies to rigid bodies /
# markers in frame data only; camera positions in data descriptions are always
# Y-up. NatNet does not transmit this setting, so it is fixed here. If Motive
# is ever switched back to its factory default (Y-up), set this to "y" —
# otherwise obstacles appear rotated 90° relative to the floor.
DEFAULT_STREAM_UP_AXIS = "z"

# Quality threshold: warn if mean marker error exceeds this (metres)
_ERROR_THRESHOLD_M = 0.005

# How many frames to accumulate for capture_frame() before picking best
_CAPTURE_N_FRAMES  = 10
_CAPTURE_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# NatNetConnection
# ---------------------------------------------------------------------------

class NatNetConnection:
    """
    Thin, correct wrapper around the NatNet 4.1 SDK.

    The SDK works as follows:
      - sdk.run()           opens sockets and starts receiver threads
      - sdk.shutdown()      closes sockets and joins threads
      - sdk.connected()     returns True once server handshake is complete
      - sdk.rigid_body_listener = fn(id, pos, rot)   called per body per frame
      - sdk.new_frame_listener  = fn(data_dict)       called once per frame
        data_dict keys: frame_number, timestamp, is_recording,
                        rigid_body_count, labeled_marker_count, ...
        NOTE: data_dict does NOT contain positions — only counts + metadata.

    Marker positions come from labeled_marker_data inside the full MoCapData
    object. The SDK does not pass MoCapData to new_frame_listener, so we
    capture it by storing the last parsed mocap_data object via a subclass.

    Camera positions come from DataDescriptions, requested via
    NAT_REQUEST_MODELDEF after run().
    """

    def __init__(
        self,
        server_ip:         str  = DEFAULT_SERVER_IP,
        client_ip:         str | None = DEFAULT_CLIENT_IP,
        use_multicast:     bool = DEFAULT_USE_MULTICAST,
        stream_up_axis:    str  = DEFAULT_STREAM_UP_AXIS,
        on_frame_received: Callable[[Frame], None] | None = None,
    ) -> None:
        self.server_ip     = server_ip
        self.client_ip     = client_ip
        self.use_multicast = use_multicast
        if stream_up_axis not in ("y", "z"):
            raise ValueError(f"stream_up_axis must be 'y' or 'z', got {stream_up_axis!r}")
        self.stream_up_axis = stream_up_axis
        self._on_frame_cb  = on_frame_received

        self._sdk: _NatNetSDK | None = None
        self._running = False

        # Populated by _on_rigid_body — latest pose per body id
        self._rb_poses:  dict[int, tuple] = {}   # id → (pos, rot, error, valid)
        self._rb_lock    = threading.Lock()

        # Populated by _on_new_frame — latest frame metadata
        self._frame_meta: dict = {}
        self._frame_event = threading.Event()

        # Labeled markers: model_id → list of (pos, residual)
        # Updated each frame via our MoCapData intercept
        self._labeled_markers: dict[int, list] = {}
        self._markers_lock = threading.Lock()

        # Body name registry: id → name  (from DataDescriptions)
        self._id_to_name: dict[int, str] = {}
        self._names_received = threading.Event()

        # Camera list (from DataDescriptions)
        self._cameras: list[CameraInfo] = []
        self._cameras_received = threading.Event()

        # Buffer for capture_frame()
        self._capture_buffer: list[Frame] = []
        self._capture_lock   = threading.Lock()
        self._capturing      = False

        self._latest_frame: Frame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Start the NatNet connection.
        Returns True if the SDK started successfully.
        Note: connected() may still return False for a few hundred ms
        while the server handshake completes — use wait_until_connected().
        """
        if not _SDK_AVAILABLE:
            logger.error(
                f"Could not import the NatNet SDK ({_SDK_IMPORT_ERROR}). "
                "Make sure NatNetClient.py, DataDescriptions.py and MoCapData.py "
                "from the OptiTrack SDK are in src/obstacle_builder/natnet/."
            )
            return False

        self._sdk = _NatNetSDKInterceptor(self)

        # Configure before run() — settings are locked after run()
        self._sdk.server_ip_address = self.server_ip
        if self.client_ip is None:
            self.client_ip = _local_ip_for(self.server_ip)
        self._sdk.local_ip_address  = self.client_ip
        self._sdk.use_multicast     = self.use_multicast
        self._sdk.print_level       = 0   # silence SDK stdout

        # Register callbacks
        self._sdk.rigid_body_listener = self._on_rigid_body
        self._sdk.new_frame_listener  = self._on_new_frame

        ok = self._sdk.run()
        if not ok:
            logger.error("NatNet sdk.run() failed — check IP addresses and that Motive is streaming.")
            return False

        self._running = True
        logger.info(f"NatNet started — server={self.server_ip}, local={self.client_ip}, "
                    f"multicast={self.use_multicast}, stream up axis={self.stream_up_axis.upper()}")

        # Wait briefly for server handshake
        if not self._wait_until_connected(timeout=5.0):
            logger.error("Timed out waiting for server handshake. Check network and Motive streaming settings.")
            self.disconnect()
            return False

        logger.info(f"Connected. App: {self._sdk.get_application_name()}, "
                    f"NatNet version: {self._sdk.get_nat_net_requested_version()}")

        # Request data descriptions to get rigid body names and camera positions
        self._sdk.send_request(
            self._sdk.command_socket,
            self._sdk.NAT_REQUEST_MODELDEF,
            "",
            (self.server_ip, self._sdk.command_port),
        )

        # Give descriptions time to arrive
        self._names_received.wait(timeout=3.0)
        self._cameras_received.wait(timeout=3.0)

        if not self._id_to_name:
            logger.warning(
                "No rigid body names received. "
                "Check that rigid bodies are defined in Motive."
            )
        if not self._cameras:
            logger.warning("No camera descriptions received.")

        return True

    def disconnect(self) -> None:
        """Shut down the NatNet connection cleanly."""
        if self._sdk and self._running:
            self._sdk.shutdown()
            self._running = False
            logger.info("NatNet disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._running and (self._sdk is not None) and self._sdk.connected()

    def capture_frame(self, num_frames: int = _CAPTURE_N_FRAMES) -> Frame | None:
        """
        Collect num_frames consecutive frames and return the best one
        (lowest combined mean marker error across all rigid bodies).

        Blocks for up to _CAPTURE_TIMEOUT_S seconds.
        Returns None if not connected or no frames arrive.
        """
        if not self._running:
            logger.error("capture_frame() called but not connected.")
            return None

        logger.info(f"Capturing {num_frames} frames...")

        with self._capture_lock:
            self._capture_buffer.clear()
            self._capturing = True

        deadline = time.monotonic() + _CAPTURE_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._capture_lock:
                if len(self._capture_buffer) >= num_frames:
                    break
            time.sleep(0.01)

        with self._capture_lock:
            self._capturing = False
            buffer = list(self._capture_buffer)

        if not buffer:
            logger.error("No frames received during capture window.")
            return None

        best = self._select_best_frame(buffer)
        best.cameras = list(self._cameras)

        logger.info(
            f"Captured frame #{best.frame_number} — "
            f"{best.rigid_body_count} bodies, "
            f"{len(best.cameras)} cameras"
        )
        self._warn_poor_tracking(best)
        return best

    @property
    def latest_frame(self) -> Frame | None:
        """Most recently received frame. Updated continuously."""
        return self._latest_frame

    def set_live_callback(self, cb: Callable[[Frame], None] | None) -> None:
        self._on_frame_cb = cb

    # ------------------------------------------------------------------
    # SDK callbacks — called on the SDK receiver thread
    # ------------------------------------------------------------------

    def _on_rigid_body(self, body_id: int, pos: tuple, rot: tuple) -> None:
        """
        Called by rigid_body_listener for each rigid body each frame.
        pos = (x, y, z), rot = (qx, qy, qz, qw)
        Note: error and tracking_valid come from the MoCapData intercept.
        We store pos/rot here and merge with error/valid in _build_frame().
        """
        with self._rb_lock:
            existing = self._rb_poses.get(body_id, (None, None, 0.0, False))
            self._rb_poses[body_id] = (pos, rot, existing[2], existing[3])

    def _on_new_frame(self, data_dict: dict) -> None:
        """
        Called by new_frame_listener once per frame.
        data_dict contains frame metadata only — not positions.
        Positions come from _on_mocap_data() via the SDK interceptor.
        """
        self._frame_meta = data_dict
        self._frame_event.set()

        frame = self._build_frame(data_dict)
        if frame is None:
            return

        self._latest_frame = frame

        if self._capturing:
            with self._capture_lock:
                self._capture_buffer.append(frame)

        if self._on_frame_cb:
            try:
                self._on_frame_cb(frame)
            except Exception:
                logger.exception("Exception in frame callback")

    def _on_mocap_data(self, mocap_data) -> None:
        """
        Called by our SDK interceptor with the full MoCapData object
        after each frame is parsed. This is where we get:
          - Per-body error and tracking_valid
          - Labeled marker positions (keyed by model_id)
        """
        # Update error + tracking_valid for each rigid body
        for rb in mocap_data.rigid_body_data.rigid_body_list:
            with self._rb_lock:
                existing = self._rb_poses.get(rb.id_num, (rb.pos, rb.rot, 0.0, False))
                self._rb_poses[rb.id_num] = (
                    existing[0],
                    existing[1],
                    rb.error,
                    rb.tracking_valid,
                )

        # Update labeled markers grouped by model_id
        new_markers: dict[int, list] = {}
        for lm in mocap_data.labeled_marker_data.labeled_marker_list:
            model_id = lm.id_num >> 16
            if model_id not in new_markers:
                new_markers[model_id] = []
            new_markers[model_id].append((lm.pos, lm.residual))

        with self._markers_lock:
            self._labeled_markers = new_markers

    def _on_data_descriptions(self, data_descs) -> None:
        """
        Called by our SDK interceptor when DataDescriptions arrive.
        Populates the id→name registry and camera list.
        """
        # Rigid body names
        for rb_desc in data_descs.rigid_body_list:
            name = rb_desc.sz_name
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            self._id_to_name[rb_desc.id_num] = name
            logger.debug(f"Registered body: id={rb_desc.id_num} name='{name}'")
        if self._id_to_name:
            self._names_received.set()

        # Cameras
        cameras = []
        for cam in data_descs.camera_list:
            name = cam.name
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            pos = cam.position   # [x, y, z]
            cameras.append(CameraInfo(
                camera_id = len(cameras),
                name      = name,
                position  = Vector3(pos[0], pos[1], pos[2]),
            ))
        if cameras:
            self._cameras = cameras
            self._cameras_received.set()
            logger.info(f"Received {len(cameras)} cameras from data descriptions.")

    # ------------------------------------------------------------------
    # Frame assembly
    # ------------------------------------------------------------------

    def _build_frame(self, meta: dict) -> Frame | None:
        """Assemble a Frame dataclass from the latest cached SDK state."""
        try:
            with self._rb_lock:
                rb_snapshot = dict(self._rb_poses)
            with self._markers_lock:
                marker_snapshot = dict(self._labeled_markers)

            rigid_bodies = []
            for body_id, (pos, rot, error, valid) in rb_snapshot.items():
                name = self._id_to_name.get(body_id, f"rigid_body_{body_id}")

                # Collect labeled markers belonging to this body
                markers = []
                for i, (mpos, residual) in enumerate(marker_snapshot.get(body_id, [])):
                    markers.append(Marker(
                        marker_id = i,
                        position  = Vector3(mpos[0], mpos[1], mpos[2]),
                        residual  = residual,
                    ))

                body = RigidBody(
                    body_id        = body_id,
                    name           = name,
                    position       = Vector3(pos[0], pos[1], pos[2]),
                    orientation    = Quaternion(rot[0], rot[1], rot[2], rot[3]),
                    markers        = markers,
                    tracking_valid = valid,
                    mean_error     = error,
                )
                # Frames are always returned in native Y-up (like the cameras).
                if self.stream_up_axis == "z":
                    body = zup_stream_to_native(body)
                rigid_bodies.append(body)

            return Frame(
                frame_number = meta.get("frame_number", 0),
                timestamp    = meta.get("timestamp", 0.0),
                rigid_bodies = rigid_bodies,
                cameras      = [],
                is_recording = meta.get("is_recording", False),
            )
        except Exception:
            logger.exception("Error building frame")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_until_connected(self, timeout: float = 5.0) -> bool:
        """Poll sdk.connected() until True or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._sdk and self._sdk.connected():
                return True
            time.sleep(0.1)
        return False

    @staticmethod
    def _select_best_frame(frames: list[Frame]) -> Frame:
        """Return the frame with the lowest average mean_error across bodies."""
        def score(f: Frame) -> float:
            valid = [b for b in f.rigid_bodies if b.tracking_valid]
            if not valid:
                return float("inf")
            return sum(b.mean_error for b in valid) / len(valid)
        return min(frames, key=score)

    @staticmethod
    def _warn_poor_tracking(frame: Frame) -> None:
        for body in frame.rigid_bodies:
            if not body.tracking_valid:
                logger.warning(
                    f"'{body.name}' (id={body.body_id}): tracking invalid — "
                    "markers may be occluded."
                )
            elif body.mean_error > _ERROR_THRESHOLD_M:
                logger.warning(
                    f"'{body.name}' (id={body.body_id}): "
                    f"mean error {body.mean_error * 1000:.1f} mm "
                    f"(threshold {_ERROR_THRESHOLD_M * 1000:.0f} mm)."
                )


# ---------------------------------------------------------------------------
# SDK interceptor — subclass that hooks into the parsed data stream
# ---------------------------------------------------------------------------

class _NatNetSDKInterceptor(_NatNetSDK if _SDK_AVAILABLE else object):
    """
    Subclass of NatNetClient that overrides the internal unpack methods
    to give us access to the full MoCapData and DataDescriptions objects
    that the SDK parses but does not expose via listeners.
    """

    def __init__(self, owner: NatNetConnection) -> None:
        if _SDK_AVAILABLE:
            super().__init__()
        self._owner = owner

    # NOTE on naming: the SDK defines these as private double-underscore
    # methods, which Python name-mangles to `_NatNetClient__<name>`. The SDK
    # calls them via `self.__unpack_...`, i.e. `self._NatNetClient__unpack_...`.
    # To override them we must use the mangled name explicitly — a plain
    # `def __unpack_mocap_data` here would become
    # `_NatNetSDKInterceptor__unpack_mocap_data` and never be called.
    # tests/test_client.py guards against this regressing.

    # The SDK binds its multicast data socket to (local_ip, port). That works
    # on Windows, but on Linux/macOS a socket bound to a unicast address never
    # receives datagrams addressed to a multicast group, so no frames arrive.
    # Bind to all interfaces instead; the group membership stays on local_ip,
    # so we still only receive Motive's stream from that interface.
    def _NatNetClient__create_data_socket(self, port):
        if not self.use_multicast or sys.platform == "win32":
            return super()._NatNetClient__create_data_socket(port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                socket.inet_aton(self.multicast_address)
                + socket.inet_aton(self.local_ip_address),
            )
            sock.bind(("", port))
        except OSError:
            logger.exception(
                f"Could not open NatNet multicast data socket "
                f"({self.multicast_address}:{port} on {self.local_ip_address})"
            )
            sock.close()
            return None
        return sock

    # Multicast frames start arriving as soon as the data socket opens, before
    # Motive has answered NAT_CONNECT. Until then the SDK's NatNet version is
    # 0.0, so it parses those frames with the wrong layout ("Marker count too
    # high", "Early End of Data Frame Suffix Data"). Drop them instead.
    def _NatNetClient__process_message(self, data, print_level=0):
        if (self.get_major() == 0
                and int.from_bytes(data[0:2], "little") == self.NAT_FRAMEOFDATA):
            return self.NAT_FRAMEOFDATA
        return super()._NatNetClient__process_message(data, print_level)

    # The SDK calls this internally after parsing each frame.
    # We override it to intercept the mocap_data object.
    def _NatNetClient__unpack_mocap_data(self, data, packet_size, major, minor):
        offset, mocap_data = super()._NatNetClient__unpack_mocap_data(
            data, packet_size, major, minor
        )
        # Never let our handler crash the SDK's receive thread, but do report it.
        try:
            self._owner._on_mocap_data(mocap_data)
        except Exception:
            logger.exception("Error handling NatNet mocap frame")
        return offset, mocap_data

    # The SDK calls this after parsing DataDescriptions.
    def _NatNetClient__unpack_data_descriptions(self, data, packet_size, major, minor):
        data = bytes(data)
        consumed = len(data)
        try:
            if _has_description_sizes(major, minor):
                # Work around a bug in OptiTrack's Python sample (NatNet SDK
                # 4.1): see _strip_description_sizes().
                data, consumed = _strip_description_sizes(data)
            _, data_descs = super()._NatNetClient__unpack_data_descriptions(
                data, packet_size, major, minor
            )
        except Exception:
            # Keep the SDK's command thread alive; without descriptions we
            # just lack body names / camera positions.
            logger.exception(
                f"NatNet SDK failed to parse data descriptions (stream NatNet "
                f"{major}.{minor}). The Python SDK may not match Motive's version."
            )
            return consumed, _SDKDataDescriptions.DataDescriptions()
        try:
            self._owner._on_data_descriptions(data_descs)
        except Exception:
            logger.exception("Error handling NatNet data descriptions")
        return consumed, data_descs


def _local_ip_for(server_ip: str) -> str:
    """
    Return the IP of the local interface the OS would use to reach server_ip.
    Connecting a UDP socket sends no packets; it only selects a route.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((server_ip, 1510))
            return s.getsockname()[0]
        except OSError:
            logger.warning(f"No route to {server_ip}; listening on all interfaces.")
            return "0.0.0.0"


def _has_description_sizes(major: int, minor: int) -> bool:
    """NatNet >= 4.1 prefixes every data description with its size in bytes."""
    return (major == 4 and minor >= 1) or major > 4


def _strip_description_sizes(data: bytes) -> tuple[bytes, int]:
    """
    Convert a NatNet >= 4.1 data-descriptions payload to the layout the
    vendored Python SDK parser expects.

    Wire format (per OptiTrack's C++ PacketClient.cpp, UnpackDescription):
        int32 count, then per description: int32 type, int32 sizeInBytes, body
    The SDK's Python parser (NatNetClient.py 4.1) omits sizeInBytes, so it
    reads the size as the start of the body and misaligns (e.g.
    UnicodeDecodeError on the rigid-body name). We drop the size fields and
    use them to cut each body out exactly.

    Returns (payload without size fields, number of input bytes consumed).
    """
    count = int.from_bytes(data[0:4], "little")
    out = bytearray(data[0:4])
    offset = 4
    for _ in range(count):
        if offset + 8 > len(data):
            raise ValueError("Truncated NatNet data-descriptions packet")
        type_bytes = data[offset:offset + 4]
        size = int.from_bytes(data[offset + 4:offset + 8], "little")
        offset += 8
        if size < 0 or offset + size > len(data):
            raise ValueError(f"Bad NatNet description size {size} at offset {offset - 4}")
        out += type_bytes
        out += data[offset:offset + size]
        offset += size
    return bytes(out), offset


# ---------------------------------------------------------------------------
# Mock connection — for development without OptiTrack hardware
# ---------------------------------------------------------------------------

class MockNatNetConnection:
    """
    Drop-in replacement for NatNetConnection.
    Loads data from tests/fixtures/sample_scene.json instead of
    connecting to OptiTrack. Identical public API.

    Usage:
        client = MockNatNetConnection("tests/fixtures/sample_scene.json")
        client.connect()
        frame = client.capture_frame()
    """

    def __init__(self, fixture_path: str = "tests/fixtures/sample_scene.json") -> None:
        import json
        from pathlib import Path
        with open(Path(fixture_path)) as f:
            self._data = json.load(f)
        self._connected = False
        self._frame: Frame | None = None

    def connect(self) -> bool:
        self._frame = self._build_frame()
        self._connected = True
        logger.info("MockNatNetConnection: fixture loaded.")
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def capture_frame(self, num_frames: int = 1) -> Frame | None:
        return self._frame

    @property
    def latest_frame(self) -> Frame | None:
        return self._frame

    def set_live_callback(self, cb) -> None:
        pass

    def _build_frame(self) -> Frame:
        rigid_bodies = []
        for rb in self._data.get("rigid_bodies", []):
            markers = [
                Marker(marker_id=i, position=Vector3(*m))
                for i, m in enumerate(rb.get("markers", []))
            ]
            p, q = rb["position"], rb["quaternion"]
            rigid_bodies.append(RigidBody(
                body_id        = rb["id"],
                name           = rb["name"],
                position       = Vector3(*p),
                orientation    = Quaternion(*q),
                markers        = markers,
                tracking_valid = True,
                mean_error     = 0.0,
            ))

        cameras = [
            CameraInfo(
                camera_id = c["id"],
                name      = f"Camera {c['id']}",
                position  = Vector3(*c["position"]),
                is_corner = c.get("is_corner", False),
            )
            for c in self._data.get("cameras", [])
        ]

        return Frame(
            frame_number = 1,
            timestamp    = 0.0,
            rigid_bodies = rigid_bodies,
            cameras      = cameras,
        )