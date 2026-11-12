"""
GUI behaviour tests (headless: Qt offscreen platform). No hardware.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from obstacle_builder.gui.main_window import MainWindow
from obstacle_builder.scene.scene_writer import build_session, export_session


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app):
    w = MainWindow()
    yield w
    w._clear_scene()
    w.deleteLater()


def test_mock_scene_is_a_toggle(win):
    win._act_mock.trigger()
    assert win._act_mock.isChecked()
    assert len(win._hulls) == 3
    win._act_mock.trigger()
    assert not win._act_mock.isChecked()
    assert win._hulls == {}
    assert win._client is None


def test_disconnect_unticks_mock(win):
    win._act_mock.trigger()
    assert win._btn_disconnect.isEnabled()
    win._btn_disconnect.click()
    assert not win._act_mock.isChecked()
    assert win._hulls == {}


def _export_mock(win, tmp_path):
    win._act_mock.trigger()
    session = build_session(win._hulls, win._boundary, win._world_frame.rigid_bodies)
    return export_session(session, tmp_path)


def test_open_session_is_read_only_view(win, tmp_path):
    session_dir = _export_mock(win, tmp_path)
    assert win.open_session(session_dir)
    assert not win._act_mock.isChecked()                   # mock replaced
    assert len(win._hulls) == 3
    assert not win._btn_export.isEnabled()
    assert not win._boundary_group.isEnabled()
    assert not win._convention_group.isEnabled()
    assert session_dir.name in win._connection_label.text()
    win._btn_disconnect.click()                            # closes the session
    assert win._session is None
    assert win._boundary_group.isEnabled()


def test_mock_replaces_open_session(win, tmp_path):
    session_dir = _export_mock(win, tmp_path)
    win.open_session(session_dir)
    win._act_mock.trigger()
    assert win._session is None
    assert win._act_mock.isChecked()
    assert win._btn_export.isEnabled()


def test_open_session_bad_path_keeps_current_scene(win, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)
    win._act_mock.trigger()
    assert not win.open_session(tmp_path / "nope")
    assert win._act_mock.isChecked()
    assert len(win._hulls) == 3


def test_rename_keeps_pose_in_viewport(win):
    win._act_mock.trigger()
    old = next(iter(win._hulls))
    pose = win._viewport._poses[old]
    win._on_obstacle_renamed(old, "renamed")
    assert "renamed" in win._viewport._poses
    assert win._viewport._poses["renamed"] == pose


@pytest.mark.parametrize("with_gazebo", [False, True])
def test_export_optionally_writes_gazebo_world(win, tmp_path, monkeypatch, with_gazebo):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    win._act_mock.trigger()
    win._chk_gazebo.setChecked(with_gazebo)
    win._on_export()
    session_dir = next(tmp_path.glob("session_*"))
    assert (session_dir / "scene.json").exists()
    assert (session_dir / "world.sdf").exists() == with_gazebo


def test_axis_option_hidden_and_z_up(win, tmp_path, monkeypatch):
    assert win._convention_group.isHidden()          # isVisible() is False for any unshown window
    assert win._radio_zup.isChecked()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    win._act_mock.trigger()
    win._on_export()
    import json
    scene = json.loads(next(tmp_path.glob("session_*/scene.json")).read_text())
    assert scene["convention"] == "z_up"
