"""
GUI behaviour tests (headless: Qt offscreen platform). No hardware.
"""

from __future__ import annotations

import os
from pathlib import Path

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


def test_export_optionally_writes_octomap(win, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    win._act_mock.trigger()
    win._chk_octomap.setChecked(True)
    win._on_export()
    assert (next(tmp_path.glob("session_*")) / "octomap.bt").exists()


# ---------------------------------------------------------------------------
# Reconstruct mode (with the mock scene as the "live" scene)
# ---------------------------------------------------------------------------

def _shifted_session(win, tmp_path, *, shift=0.0, rename=None):
    """
    A session exported from the mock scene, edited to stand for a setup where
    the first obstacle was elsewhere (shift, metres in x) or had another name.
    The mock's live poses never change, so editing the target is the test.
    """
    import json
    session_dir = _export_mock(win, tmp_path)
    scene_path = session_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["models"][0]["position"][0] += shift
    if rename:
        scene["models"][0]["name"] = rename
    scene_path.write_text(json.dumps(scene))
    return session_dir, scene["models"][0]["name"]


def test_reconstruct_needs_a_connection(win, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    session_dir = _export_mock(win, tmp_path)
    win._clear_scene()
    assert not win._act_reconstruct.isEnabled()
    assert not win.start_reconstruction(session_dir)
    win.open_session(session_dir)                         # a saved session is not live
    assert not win._act_reconstruct.isEnabled()


def test_reconstruct_all_in_place(win, tmp_path):
    session_dir = _export_mock(win, tmp_path)
    assert win._act_reconstruct.isEnabled()
    assert win.start_reconstruction(session_dir)
    assert not win._recon_group.isHidden()
    assert win._obstacles_group.isHidden() and win._export_group.isHidden()
    assert not win._btn_refresh.isEnabled() and not win._btn_export.isEnabled()
    assert win._recon_summary.text() == "All 3 in place"
    assert set(win._viewport._targets) == set(win._hulls)
    assert win._live_timer.isActive()


def test_reconstruct_shows_moved_obstacle_red(win, tmp_path):
    from obstacle_builder.gui.placement_table import STATUS_COLOURS
    from obstacle_builder.reconstruct import State
    session_dir, name = _shifted_session(win, tmp_path, shift=0.10)
    win.start_reconstruction(session_dir)
    assert win._placements[name].state is State.MISPLACED
    assert win._recon_summary.text() == "2 / 3 in place"
    assert win._viewport._status_colours[name] == STATUS_COLOURS[State.MISPLACED]
    row = win._placement_table._names.index(name)
    assert win._placement_table.item(row, 2).text() == "10.0"
    assert "+x 10 cm" in win._placement_table.item(row, 2).toolTip()   # move it back to +x


def test_reconstruct_tolerance_is_live(win, tmp_path):
    from obstacle_builder.reconstruct import State
    session_dir, name = _shifted_session(win, tmp_path, shift=0.03)
    win.start_reconstruction(session_dir)
    assert win._placements[name].state is State.MISPLACED          # 3 cm > 2 cm
    win._recon_pos_tol.setValue(5.0)
    assert win._placements[name].state is State.PLACED
    assert win._recon_summary.text() == "All 3 in place"


def test_reconstruct_missing_obstacle_drawn_as_target_only(win, tmp_path):
    from obstacle_builder.reconstruct import State
    session_dir, name = _shifted_session(win, tmp_path, rename="not_streamed")
    win.start_reconstruction(session_dir)
    assert win._placements[name].state is State.MISSING
    assert name in win._viewport._targets and name not in win._viewport._poses
    row = win._placement_table._names.index(name)
    assert win._placement_table.item(row, 2).text() == "—"


def test_reconstruct_shows_live_obstacle_not_in_session(win, tmp_path):
    """Renaming in the session leaves the live body unmatched: shown amber, listed."""
    from obstacle_builder.gui.placement_table import STATUS_COLOURS
    from obstacle_builder.reconstruct import State
    session_dir, _ = _shifted_session(win, tmp_path, rename="not_streamed")
    win.start_reconstruction(session_dir)
    extra = [n for n, p in win._placements.items() if p.state is State.EXTRA]
    assert len(extra) == 1
    name = extra[0]                                    # the renamed obstacle's live body
    assert name in win._viewport._hulls and name in win._viewport._poses
    assert name not in win._viewport._targets
    assert win._viewport._status_colours[name] == STATUS_COLOURS[State.EXTRA]
    assert win._placement_table._names[-1] == name
    assert win._recon_summary.text() == "2 / 3 in place · 1 extra"


def test_reconstruct_demo_example_shows_every_state(win):
    """examples/reconstruct_demo (README) against the mock scene."""
    from obstacle_builder.reconstruct import State
    demo = Path(__file__).parents[1] / "examples" / "reconstruct_demo"
    win._act_mock.trigger()
    assert win.start_reconstruction(demo / "scene.json")
    assert {n: p.state for n, p in win._placements.items()} == {
        "wall_left": State.MISPLACED, "crate": State.MISSING,
        "ramp_a": State.PLACED, "pillar_1": State.EXTRA,
    }
    assert win._recon_summary.text() == "1 / 3 in place · 1 extra"


def test_reconstruct_stop_and_disconnect(win, tmp_path):
    session_dir = _export_mock(win, tmp_path)
    win.start_reconstruction(session_dir)
    win._btn_recon_stop.click()
    assert win._recon_session is None and win._viewport._targets == {}
    assert win._recon_group.isHidden() and not win._obstacles_group.isHidden()
    assert len(win._hulls) == 3 and win._btn_export.isEnabled()
    assert not win._live_timer.isActive()                           # mock: no live updates
    win.start_reconstruction(session_dir)
    win._btn_disconnect.click()
    assert win._recon_session is None and win._recon_group.isHidden()


# ---------------------------------------------------------------------------
# Viewport navigation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 180.0, -135.0])
def test_pan_moves_the_scene_with_the_mouse(azimuth):
    """Whichever way the camera faces, a drag right/down moves the scene right/down."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from obstacle_builder.gui.viewport import _pan_step

    elevation = 30.0
    # paintGL: eye = Rx(-elevation) · Rz(azimuth) · (world - pan)
    view = (Rotation.from_euler("x", -elevation, degrees=True)
            * Rotation.from_euler("z", azimuth, degrees=True))

    def on_screen(pan):            # where the world origin appears (eye x right, eye y up)
        return view.apply(-np.array([*pan, 0.0]))[:2]

    start = on_screen((0.0, 0.0))
    right = on_screen(_pan_step(10, 0, azimuth, 0.01)) - start
    down  = on_screen(_pan_step(0, 10, azimuth, 0.01)) - start
    assert right[0] > 0 and right[1] == pytest.approx(0, abs=1e-9)
    assert down[1] < 0 and down[0] == pytest.approx(0, abs=1e-9)


def test_lab_logo_and_icon(win):
    from obstacle_builder.gui.main_window import RESOURCES
    assert not win.windowIcon().isNull()
    box = win._about_box()
    assert not box.iconPixmap().isNull()
    assert "Multi-Robot Systems Lab" in box.text()
    assert {p.name for p in RESOURCES.glob("*.png")} == {"mrs_icon.png", "mrs_logo.png"}


def test_help_menu_opens_the_guides(win):
    from obstacle_builder.gui.doc_viewer import GUIDES, REPO_ROOT
    for label, relative in GUIDES:
        assert (REPO_ROOT / relative).is_file(), relative
        win._open_guide(relative)
        text = win._doc_viewer._browser.toPlainText()
        first_heading = next(line for line in (REPO_ROOT / relative).read_text().splitlines()
                             if line.startswith("# "))
        assert first_heading[2:] in text, label
        assert win._doc_viewer.windowTitle()
    win._doc_viewer.close()


@pytest.mark.parametrize(("env", "expected"), [
    ({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, "x11"),
    ({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0", "GDK_BACKEND": "wayland"}, "wayland"),
    ({"WAYLAND_DISPLAY": "wayland-0"}, None),           # no XWayland: leave GTK alone
    ({"DISPLAY": ":0"}, None),                          # X11 session: nothing to do
])
def test_gtk3_wayland_workaround(monkeypatch, env, expected):
    from obstacle_builder.gui import main_window
    monkeypatch.setattr(main_window.sys, "platform", "linux")
    main_window._avoid_gtk3_wayland_crash(env)
    assert env.get("GDK_BACKEND") == expected
