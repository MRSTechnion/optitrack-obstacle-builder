"""
Tests for the NatNet SDK interceptor in obstacle_builder.natnet.client.

The interceptor overrides private (double-underscore) methods of the vendored
OptiTrack NatNetClient. Python name-mangles those, so an override only takes
effect if it uses the exact mangled name the SDK calls. These tests make sure
the hook stays wired up — no OptiTrack hardware or network needed.
"""

from __future__ import annotations

import pytest

from obstacle_builder.natnet import client as client_mod
from obstacle_builder.natnet.NatNetClient import NatNetClient
from obstacle_builder.natnet.types import Frame

HOOKED = ["_NatNetClient__unpack_mocap_data", "_NatNetClient__unpack_data_descriptions"]


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

    assert sdk._NatNetClient__unpack_data_descriptions(b"", 0, 4, 1) == (7, sentinel)
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
