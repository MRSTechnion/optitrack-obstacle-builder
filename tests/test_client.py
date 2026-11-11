"""
Tests for the NatNet SDK interceptor in obstacle_builder.natnet.client.

The interceptor overrides private (double-underscore) methods of the vendored
OptiTrack NatNetClient. Python name-mangles those, so an override only takes
effect if it uses the exact mangled name the SDK calls. These tests make sure
the hook stays wired up — no OptiTrack hardware or network needed.
"""

from __future__ import annotations

import struct

import pytest

from obstacle_builder.natnet import client as client_mod
from obstacle_builder.natnet.NatNetClient import NatNetClient
from obstacle_builder.natnet.types import Frame

HOOKED = [
    "_NatNetClient__unpack_mocap_data",
    "_NatNetClient__unpack_data_descriptions",
    "_NatNetClient__create_data_socket",
]


class _FakeOwner:
    def __init__(self, fail: bool = False):
        self.mocap = []
        self.descs = []
        self.fail = fail

    def _on_mocap_data(self, mocap_data):
        if self.fail:
            raise RuntimeError("boom")
        self.mocap.append(mocap_data)

    def _on_data_descriptions(self, data_descs):
        self.descs.append(data_descs)


@pytest.mark.parametrize("name", HOOKED)
def test_sdk_still_defines_hooked_method(name):
    """If the SDK is updated and renames these, our overrides silently die."""
    assert name in vars(NatNetClient), f"SDK no longer defines {name}"


@pytest.mark.parametrize("name", HOOKED)
def test_interceptor_overrides_sdk_method(name):
    assert name in vars(client_mod._NatNetSDKInterceptor)
    assert getattr(client_mod._NatNetSDKInterceptor, name) is not getattr(NatNetClient, name)


def test_mocap_data_forwarded_to_owner(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        NatNetClient, "_NatNetClient__unpack_mocap_data", lambda self, *a: (42, sentinel)
    )
    owner = _FakeOwner()
    sdk = client_mod._NatNetSDKInterceptor(owner)

    # Call it the way the SDK does internally: self.__unpack_mocap_data(...)
    assert sdk._NatNetClient__unpack_mocap_data(b"", 0, 4, 1) == (42, sentinel)
    assert owner.mocap == [sentinel]


def test_data_descriptions_forwarded_to_owner(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        NatNetClient, "_NatNetClient__unpack_data_descriptions", lambda self, *a: (7, sentinel)
    )
    owner = _FakeOwner()
    sdk = client_mod._NatNetSDKInterceptor(owner)

    _, descs = sdk._NatNetClient__unpack_data_descriptions(b"", 0, 4, 0)
    assert descs is sentinel
    assert owner.descs == [sentinel]


def test_owner_error_is_logged_not_raised(monkeypatch, caplog):
    sentinel = object()
    monkeypatch.setattr(
        NatNetClient, "_NatNetClient__unpack_mocap_data", lambda self, *a: (0, sentinel)
    )
    sdk = client_mod._NatNetSDKInterceptor(_FakeOwner(fail=True))

    assert sdk._NatNetClient__unpack_mocap_data(b"", 0, 4, 1) == (0, sentinel)
    assert "Error handling NatNet mocap frame" in caplog.text


def test_frame_rigid_body_count_is_property():
    frame = Frame(frame_number=1, timestamp=0.0)
    assert frame.rigid_body_count == 0


def test_sdk_importable_inside_package():
    """The vendored SDK must import as part of the package, or live mode is dead."""
    assert client_mod._SDK_AVAILABLE, client_mod._SDK_IMPORT_ERROR


# ---------------------------------------------------------------------------
# NatNet 4.1 data descriptions (per-description sizeInBytes field)
# ---------------------------------------------------------------------------

NAT_MODELDEF = 5


def _rigid_body_desc_body(name: str, body_id: int, markers: int = 3) -> bytes:
    """Rigid body description body as laid out in NatNet 4.x."""
    b = name.encode() + b"\0"
    b += struct.pack("<ii", body_id, -1)                 # id, parent id
    b += struct.pack("<fff", 0.0, 0.0, 0.0)              # position offset
    b += struct.pack("<i", markers)                      # marker count
    for i in range(markers):
        b += struct.pack("<fff", 0.1 * i, 0.0, 0.0)      # marker offsets
    for i in range(markers):
        b += struct.pack("<i", i + 1)                    # active labels
    for i in range(markers):
        b += f"m{i}".encode() + b"\0"                    # marker names
    return b


def _descriptions_payload(bodies, with_sizes: bool) -> bytes:
    p = struct.pack("<i", len(bodies))
    for name, body_id in bodies:
        body = _rigid_body_desc_body(name, body_id)
        p += struct.pack("<i", 1)                        # type 1 = rigid body
        if with_sizes:
            p += struct.pack("<i", len(body))            # NatNet >= 4.1
        p += body
    return p


BODIES = [("wall_left", 1), ("pillar_1", 2)]


def test_strip_description_sizes_matches_pre_41_layout():
    v41 = _descriptions_payload(BODIES, with_sizes=True)
    stripped, consumed = client_mod._strip_description_sizes(v41)
    assert stripped == _descriptions_payload(BODIES, with_sizes=False)
    assert consumed == len(v41)


def test_strip_description_sizes_rejects_truncated_packet():
    v41 = _descriptions_payload(BODIES, with_sizes=True)
    with pytest.raises(ValueError):
        client_mod._strip_description_sizes(v41[:-5])


@pytest.mark.parametrize(("minor", "with_sizes"), [(0, False), (1, True)])
def test_sdk_parses_modeldef_packet_end_to_end(minor, with_sizes):
    """Run a full NAT_MODELDEF packet through the SDK's own message handler."""
    owner = _FakeOwner()
    sdk = client_mod._NatNetSDKInterceptor(owner)
    sdk._NatNetClient__nat_net_requested_version = [4, minor, 0, 0]

    payload = _descriptions_payload(BODIES, with_sizes=with_sizes)
    packet = struct.pack("<HH", NAT_MODELDEF, len(payload)) + payload
    sdk._NatNetClient__process_message(packet, 0)

    assert len(owner.descs) == 1
    rbs = owner.descs[0].rigid_body_list
    assert [(rb.sz_name, rb.id_num) for rb in rbs] == [(b"wall_left", 1), (b"pillar_1", 2)]
    assert len(rbs[0].rb_marker_list) == 3


def test_unparseable_descriptions_logged_not_raised(caplog):
    owner = _FakeOwner()
    sdk = client_mod._NatNetSDKInterceptor(owner)
    sdk._NatNetClient__nat_net_requested_version = [4, 1, 0, 0]

    # 4.1 packet *without* size fields -> strip step sees nonsense sizes
    payload = _descriptions_payload(BODIES, with_sizes=False)
    packet = struct.pack("<HH", NAT_MODELDEF, len(payload)) + payload
    sdk._NatNetClient__process_message(packet, 0)

    assert "failed to parse data descriptions" in caplog.text


def test_local_ip_for_picks_routing_interface():
    assert client_mod._local_ip_for("127.0.0.1") == "127.0.0.1"


def test_default_client_ip_is_auto():
    conn = client_mod.NatNetConnection(server_ip="192.168.1.10")
    assert conn.client_ip is None  # resolved to a real NIC on connect()


def test_multicast_data_socket_binds_to_all_interfaces():
    """On Linux a socket bound to a unicast IP never sees multicast datagrams."""
    sdk = client_mod._NatNetSDKInterceptor(_FakeOwner())
    sdk.use_multicast = True
    sdk.local_ip_address = "127.0.0.1"
    sock = sdk._NatNetClient__create_data_socket(0)
    try:
        assert sock is not None
        assert sock.getsockname()[0] == "0.0.0.0"
    finally:
        sock.close()
