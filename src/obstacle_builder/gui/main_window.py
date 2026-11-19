"""
Main application window.

Layout:
    ┌─────────────────────────────┬──────────────────┐
    │  Toolbar (connect / status) │                  │
    ├─────────────────────────────┤                  │
    │                             │   Sidebar        │
    │   3D Viewport               │   - Connection   │
    │   (viewport.py)             │   - Convention   │
    │                             │   - Boundary     │
    │                             │   - Obstacles    │
    │                             │   - Export       │
    └─────────────────────────────┴──────────────────┘

Reconstruct mode (File → Reconstruct setup…, while connected): a saved
session's obstacles are shown where they belong, each live obstacle green
when it is back in place and red when not; the sidebar's Reconstruct group
replaces Obstacles and Export.

The window owns:
  - NatNetConnection (or Mock) instance
  - The latest converted Frame
  - All geometry results (hulls, boundary)
  - All GUI state

Child widgets (Viewport, ObstacleTable) communicate back via signals.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCloseEvent, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..adapters import gazebo
from ..adapters import octomap as octomap_adapter
from ..geometry.hull import BOUNDARY_BODY_NAME, build_all_hulls, build_boundary, build_hull
from ..geometry.transform import convert_frame
from ..natnet.client import DEFAULT_SERVER_IP, MockNatNetConnection, NatNetConnection
from ..natnet.types import Frame
from ..reconstruct import (
    ANGLE_TOL_DEG,
    POSITION_TOL_M,
    State,
    check_placement,
    is_complete,
    summary,
    targets_from_session,
)
from ..scene.scene_writer import build_session, export_session, load_session
from .doc_viewer import GUIDES, DocViewer, open_guide
from .obstacle_table import ObstacleTable
from .placement_table import STATUS_COLOURS, PlacementTable
from .viewport import Viewport3D

logger = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"   # logo and icon


# ---------------------------------------------------------------------------
# Worker thread — NatNet connect + capture runs off the main thread
# so the GUI never freezes
# ---------------------------------------------------------------------------

class _ConnectWorker(QThread):
    """
    Runs NatNetConnection.connect() and capture_frame() in a background
    thread. Emits signals back to the main window on completion.
    """
    connected    = pyqtSignal(bool, str)   # success, message
    frame_ready  = pyqtSignal(object)      # Frame object

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        ok = self._client.connect()
        if not ok:
            self.connected.emit(False, "Connection failed — check IP and Motive streaming.")
            return
        self.connected.emit(True, "Connected.")

        frame = self._client.capture_frame()
        if frame is None:
            self.connected.emit(False, "Connected but no frame received.")
            return
        self.frame_ready.emit(frame)


class _CaptureWorker(QThread):
    """Re-captures a fresh frame from an already-connected client."""
    frame_ready = pyqtSignal(object)
    failed      = pyqtSignal(str)

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        frame = self._client.capture_frame()
        if frame is None:
            self.failed.emit("Capture failed — no frame received.")
        else:
            self.frame_ready.emit(frame)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    # Emitted whenever geometry is rebuilt — Viewport subscribes to this
    geometry_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiTrack Obstacle Builder")
        self.setWindowIcon(QIcon(str(RESOURCES / "mrs_icon.png")))
        self.resize(1280, 800)

        # --- State ----------------------------------------------------------
        self._client: NatNetConnection | MockNatNetConnection | None = None
        self._raw_frame:   Frame | None = None   # OptiTrack Y-up frame
        self._world_frame: Frame | None = None   # converted Z-up/Y-up frame
        self._hulls  = {}          # name → HullResult
        self._skipped_bodies: list[str] = []   # streamed bodies with no hull
        self._boundary = None      # BoundaryResult or None
        # A saved session opened with File → Open session (read-only view)
        self._session = None       # LoadedSession or None
        self._z_up   = True        # current axis convention
        self._worker: QThread | None = None
        # Reconstruct mode: the saved session being rebuilt and its targets
        self._recon_session = None  # LoadedSession or None
        self._recon_targets = {}    # name → reconstruct.Target
        self._placements    = {}    # name → reconstruct.Placement (latest check)
        self._recon_extra_hulls = {}  # live obstacles not in the session: name → hull or None

        # --- Build UI -------------------------------------------------------
        self._build_ui()
        self._build_menu()
        self._update_ui_state()

        # Live-preview timer — updates viewport from latest_frame
        # while connected (10 Hz is plenty for a static scene tool)
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._on_live_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Assemble the main window layout."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Toolbar
        toolbar = self._build_toolbar()
        root_layout.addWidget(toolbar)

        # Main area: viewport + sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        self._viewport = Viewport3D(self)
        splitter.addWidget(self._viewport)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        splitter.setStretchFactor(0, 3)   # viewport gets 3× space
        splitter.setStretchFactor(1, 1)   # sidebar fixed ~280 px

        root_layout.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Not connected")
        self._status.addWidget(self._status_label)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setObjectName("toolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # Connection status indicator
        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(16)
        self._status_dot.setStyleSheet("color: #c0392b; font-size: 14px;")

        self._connection_label = QLabel("Not connected")
        self._connection_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # Server IP input
        ip_label = QLabel("OptiTrack IP:")
        self._ip_input = QLineEdit(DEFAULT_SERVER_IP)
        self._ip_input.setFixedWidth(130)
        self._ip_input.setPlaceholderText(f"e.g. {DEFAULT_SERVER_IP}")

        self._btn_connect    = QPushButton("Connect")
        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_refresh    = QPushButton("Refresh")

        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        self._btn_refresh.clicked.connect(self._on_refresh)

        layout.addWidget(self._status_dot)
        layout.addWidget(self._connection_label)
        layout.addStretch()
        layout.addWidget(ip_label)
        layout.addWidget(self._ip_input)
        layout.addWidget(self._btn_connect)
        layout.addWidget(self._btn_disconnect)
        layout.addWidget(self._btn_refresh)

        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._convention_group = self._build_convention_group()
        self._boundary_group   = self._build_boundary_group()
        layout.addWidget(self._convention_group)
        # Everything is Z-up (Motive already streams Z-up). The Y-up export
        # option is hidden, not removed: setVisible(True) brings it back.
        # Note: the boundary is built assuming Z-up, so re-enabling Y-up
        # would need build_boundary() to handle it first.
        self._convention_group.setVisible(False)
        layout.addWidget(self._boundary_group)

        # Obstacle table takes remaining space
        self._obstacles_group = QGroupBox("Obstacles")
        obs_layout = QVBoxLayout(self._obstacles_group)
        self._obstacle_table = ObstacleTable(self)
        self._obstacle_table.selection_changed.connect(self._on_obstacle_selected)
        self._obstacle_table.name_changed.connect(self._on_obstacle_renamed)
        obs_layout.addWidget(self._obstacle_table)
        layout.addWidget(self._obstacles_group, stretch=1)

        # Shown instead of Obstacles + Export while reconstructing
        self._recon_group = self._build_reconstruct_group()
        self._recon_group.setVisible(False)
        layout.addWidget(self._recon_group, stretch=1)

        self._export_group = self._build_export_group()
        layout.addWidget(self._export_group)
        return sidebar

    def _build_convention_group(self) -> QGroupBox:
        group = QGroupBox("Coordinate frame")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Axis convention:"))
        self._radio_zup = QRadioButton("Z-up")
        self._radio_yup = QRadioButton("Y-up")
        self._radio_zup.setChecked(True)
        self._radio_zup.toggled.connect(self._on_convention_changed)
        row.addWidget(self._radio_zup)
        row.addWidget(self._radio_yup)
        layout.addLayout(row)

        return group

    def _build_boundary_group(self) -> QGroupBox:
        group = QGroupBox("Room boundary")
        layout = QVBoxLayout(group)

        # Source selector
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self._boundary_source = QComboBox()
        self._boundary_source.addItem("4 corner cameras", "corners")
        self._boundary_source.addItem("All cameras", "all")
        self._boundary_source.addItem("Interest area", "rigid_body")
        self._boundary_source.setToolTip(
            "4 corner cameras: rectangular room - floor spans the 4 corner cameras.\n"
            "All cameras: any layout (circular rig, fewer cameras) - floor is the\n"
            "outline of all cameras, with one edge (and wall) per outline segment.\n"
            f"Interest area: the outline of a rigid body named '{BOUNDARY_BODY_NAME}'\n"
            "in Motive, with markers on the floor at the area's corners.\n"
            "See README / docs for details."
        )
        self._boundary_source.currentIndexChanged.connect(
            lambda _i: self._rebuild_geometry()
        )
        source_row.addWidget(self._boundary_source)
        layout.addLayout(source_row)

        # Show boundary checkbox
        self._chk_show_boundary = QCheckBox("Show boundary in preview")
        self._chk_show_boundary.setChecked(True)
        self._chk_show_boundary.toggled.connect(self._on_boundary_display_changed)
        layout.addWidget(self._chk_show_boundary)

        # Wall extrusion
        self._chk_walls = QCheckBox("Generate walls")
        self._chk_walls.toggled.connect(self._on_walls_toggled)
        layout.addWidget(self._chk_walls)

        # Wall height
        height_row = QHBoxLayout()
        height_row.addSpacing(20)
        height_row.addWidget(QLabel("Height (m):"))
        self._wall_height_spin = QDoubleSpinBox()
        self._wall_height_spin.setRange(0.5, 10.0)
        self._wall_height_spin.setValue(3.0)
        self._wall_height_spin.setSingleStep(0.1)
        self._wall_height_spin.setDecimals(1)
        self._wall_height_spin.setEnabled(False)
        self._wall_height_spin.valueChanged.connect(lambda _v: self._rebuild_geometry())
        height_row.addWidget(self._wall_height_spin)
        layout.addLayout(height_row)

        # Wall direction
        direction_row = QHBoxLayout()
        direction_row.addSpacing(20)
        self._radio_outward = QRadioButton("Outward")
        self._radio_inward  = QRadioButton("Inward")
        self._radio_outward.setChecked(True)
        self._radio_outward.setEnabled(False)
        self._radio_inward.setEnabled(False)
        # One toggled signal per switch is enough (both buttons fire); react
        # only to the one becoming checked.
        self._radio_inward.toggled.connect(lambda _c: self._rebuild_geometry())
        direction_row.addWidget(QLabel("Direction:"))
        direction_row.addWidget(self._radio_outward)
        direction_row.addWidget(self._radio_inward)
        layout.addLayout(direction_row)

        return group

    def _build_reconstruct_group(self) -> QGroupBox:
        group = QGroupBox("Reconstruct")
        layout = QVBoxLayout(group)

        self._recon_name = QLabel("")
        self._recon_name.setWordWrap(True)
        layout.addWidget(self._recon_name)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Within:"))
        self._recon_pos_tol = QDoubleSpinBox()
        self._recon_pos_tol.setRange(0.5, 50.0)
        self._recon_pos_tol.setDecimals(1)
        self._recon_pos_tol.setSingleStep(0.5)
        self._recon_pos_tol.setValue(POSITION_TOL_M * 100)
        self._recon_pos_tol.setSuffix(" cm")
        self._recon_angle_tol = QDoubleSpinBox()
        self._recon_angle_tol.setRange(0.5, 45.0)
        self._recon_angle_tol.setDecimals(1)
        self._recon_angle_tol.setSingleStep(0.5)
        self._recon_angle_tol.setValue(ANGLE_TOL_DEG)
        self._recon_angle_tol.setSuffix(" °")
        for spin in (self._recon_pos_tol, self._recon_angle_tol):
            spin.valueChanged.connect(lambda _v: self._update_reconstruction())
            tol_row.addWidget(spin)
        layout.addLayout(tol_row)

        self._placement_table = PlacementTable(self)
        self._placement_table.selection_changed.connect(self._on_obstacle_selected)
        layout.addWidget(self._placement_table, stretch=1)

        self._recon_summary = QLabel("")
        self._recon_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self._recon_summary.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self._recon_summary.setFont(font)
        layout.addWidget(self._recon_summary)

        self._btn_recon_stop = QPushButton("Stop")
        self._btn_recon_stop.clicked.connect(self._stop_reconstruction)
        layout.addWidget(self._btn_recon_stop)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        layout = QVBoxLayout(group)

        # Optional notes field
        layout.addWidget(QLabel("Notes (optional):"))
        self._notes_input = QLineEdit()
        self._notes_input.setPlaceholderText("e.g. experiment setup A")
        layout.addWidget(self._notes_input)

        self._chk_gazebo = QCheckBox("Also write Gazebo world")
        self._chk_gazebo.setToolTip(
            "Write world.sdf into the session folder (run it with: gz sim world.sdf).\n"
            "Same as running obstacle-builder-gazebo on the session afterwards."
        )
        layout.addWidget(self._chk_gazebo)

        self._chk_octomap = QCheckBox("Also write OctoMap")
        self._chk_octomap.setToolTip(
            "Write octomap.bt (5 cm 3D occupancy map) into the session folder.\n"
            "Same as running obstacle-builder-octomap on the session afterwards."
        )
        layout.addWidget(self._chk_octomap)

        self._btn_export = QPushButton("Export scene")
        self._btn_export.setFixedHeight(36)
        font = self._btn_export.font()
        font.setBold(True)
        self._btn_export.setFont(font)
        self._btn_export.clicked.connect(self._on_export)
        layout.addWidget(self._btn_export)

        self._export_status = QLabel("")
        self._export_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_status.setWordWrap(True)
        layout.addWidget(self._export_status)

        return group

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        act_open = QAction("Open session…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.setStatusTip("View a previously exported session (scene.json + STLs)")
        act_open.triggered.connect(self._on_open_session)
        file_menu.addAction(act_open)

        self._act_reconstruct = QAction("Reconstruct setup…", self)
        self._act_reconstruct.setShortcut("Ctrl+R")
        self._act_reconstruct.setStatusTip(
            "Put a saved session's obstacles back in place: green = in place, red = not"
        )
        self._act_reconstruct.triggered.connect(self._on_reconstruct)
        file_menu.addAction(self._act_reconstruct)

        act_export = QAction("Export scene…", self)
        act_export.setShortcut("Ctrl+S")
        act_export.triggered.connect(self._on_export)
        file_menu.addAction(act_export)

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Development aids — not needed for normal use
        dev_menu = menubar.addMenu("Dev")
        self._act_mock = QAction("Load mock scene", self)
        self._act_mock.setCheckable(True)
        self._act_mock.setStatusTip(
            "Toggle built-in sample data (3 obstacles, 4 cameras) to try the tool "
            "without OptiTrack. Not a saved session."
        )
        self._act_mock.toggled.connect(self._on_mock_toggled)
        dev_menu.addAction(self._act_mock)

        help_menu = menubar.addMenu("Help")
        self._doc_viewer = None
        for label, relative in GUIDES:
            act = QAction(label, self)
            act.triggered.connect(lambda _c=False, r=relative: self._open_guide(r))
            help_menu.addAction(act)
        help_menu.addSeparator()
        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # UI state management
    # ------------------------------------------------------------------

    def _update_ui_state(self):
        """Enable/disable widgets based on connection and data state."""
        mock        = isinstance(self._client, MockNatNetConnection)
        live        = self._client is not None and not mock
        viewing     = self._session is not None
        recon       = self._recon_session is not None
        has_hulls   = bool(self._hulls)
        worker_busy = self._worker is not None and self._worker.isRunning()

        # Connect also replaces a mock scene / opened session
        self._btn_connect.setEnabled(not live and not worker_busy)
        self._btn_disconnect.setEnabled(live or mock or viewing)
        self._btn_refresh.setEnabled(live and not worker_busy and not recon)
        self._ip_input.setEnabled(not live)
        # A saved session has no markers: nothing to recompute or re-export
        self._btn_export.setEnabled(has_hulls and not viewing and not recon)
        self._convention_group.setEnabled(not viewing and not recon)
        self._boundary_group.setEnabled(not viewing and not recon)
        # Reconstruct needs live poses (Motive, or the mock scene for testing)
        self._act_reconstruct.setEnabled((live or mock) and not worker_busy)
        self._obstacles_group.setVisible(not recon)
        self._export_group.setVisible(not recon)
        self._recon_group.setVisible(recon)

        # Status dot colour: live green, mock amber, saved session blue
        colour = "#27ae60" if live else "#f39c12" if mock else "#2e86de" if viewing else "#c0392b"
        self._status_dot.setStyleSheet(f"color: {colour}; font-size: 14px;")

    def _set_status(self, message: str):
        self._status_label.setText(message)
        logger.info(message)

    # ------------------------------------------------------------------
    # Slots — toolbar buttons
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_connect(self):
        ip = self._ip_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing IP", "Enter the OptiTrack PC IP address.")
            return

        self._clear_scene()   # drop a mock scene / opened session first
        self._client = NatNetConnection(server_ip=ip)
        self._set_status(f"Connecting to {ip}…")
        self._start_worker(_ConnectWorker(self._client))

    @pyqtSlot()
    def _on_disconnect(self):
        self._clear_scene("Disconnected.")

    def _clear_scene(self, status: str | None = None):
        """
        Return to the empty state: disconnect any client (live or mock), close
        an opened session, clear the views, untick Dev → Load mock scene.
        """
        self._live_timer.stop()
        self._end_reconstruction()
        if self._client:
            self._client.disconnect()
            self._client = None
        self._raw_frame      = None
        self._world_frame    = None
        self._hulls          = {}
        self._skipped_bodies = []
        self._boundary       = None
        self._session        = None
        self._obstacle_table.clear()
        self._viewport.clear()
        self._connection_label.setText("Not connected")
        self._act_mock.blockSignals(True)
        self._act_mock.setChecked(False)
        self._act_mock.blockSignals(False)
        if status:
            self._set_status(status)
        self._update_ui_state()

    @pyqtSlot()
    def _on_refresh(self):
        if not self._client:
            return
        self._set_status("Capturing fresh frame…")
        self._start_worker(_CaptureWorker(self._client))

    @pyqtSlot(bool)
    def _on_mock_toggled(self, checked: bool):
        """Dev → Load mock scene is a toggle: ticked = mock scene shown."""
        if checked:
            self._on_load_mock()
        else:
            self._clear_scene("Mock scene removed.")

    @pyqtSlot()
    def _on_load_mock(self):
        """Load the built-in mock scene (test fixture) for offline development."""
        self._clear_scene()
        fixture = Path(__file__).parents[3] / "tests" / "fixtures" / "sample_scene.json"
        if not fixture.exists():
            QMessageBox.warning(
                self, "Mock scene not found",
                f"Could not find:\n{fixture}"
            )
            self._update_ui_state()
            return
        self._client = MockNatNetConnection(str(fixture))
        self._client.connect()
        frame = self._client.capture_frame()
        if frame:
            self._on_frame_ready(frame)
            self._connection_label.setText("Mock scene loaded")
            self._status_dot.setStyleSheet("color: #f39c12; font-size: 14px;")  # amber
            self._set_status("Loaded mock scene (built-in sample data, offline).")
            self._act_mock.blockSignals(True)
            self._act_mock.setChecked(True)
            self._act_mock.blockSignals(False)
        self._update_ui_state()

    @pyqtSlot()
    def _on_open_session(self):
        """File → Open session: show a previously exported session (read-only)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session — choose its scene.json", str(Path.home()),
            "Session (scene.json);;JSON files (*.json)",
        )
        if path:
            self.open_session(Path(path))

    def open_session(self, path: Path) -> bool:
        """Load and display a saved session (directory or scene.json)."""
        try:
            session = load_session(path)
        except (OSError, ValueError) as e:
            logger.exception("Could not open session")
            QMessageBox.critical(self, "Could not open session", str(e))
            return False

        self._clear_scene()
        self._session = session
        self._hulls   = dict(session.hulls)
        self._boundary = session.boundary
        self._obstacle_table.update_from_hulls(
            self._hulls, session.bodies, marker_counts=session.marker_counts,
        )
        self._viewport.update_scene(
            self._hulls,
            self._boundary,
            show_boundary = self._chk_show_boundary.isChecked(),
            bodies        = session.bodies,
            cameras       = self._boundary.corner_cameras if self._boundary else [],
        )
        name = session.session_dir.name
        self._connection_label.setText(f"Viewing session: {name}")
        n = len(self._hulls)
        note = ""
        if session.scene.get("convention") == "y_up":
            note = " — saved Y-up, shown as stored (preview assumes Z-up)"
        self._set_status(
            f"Opened {name} — {n} obstacle{'' if n == 1 else 's'} (read-only){note}"
        )
        self._update_ui_state()
        return True

    # ------------------------------------------------------------------
    # Reconstruct mode
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_reconstruct(self):
        """File → Reconstruct setup: choose the session to put back in place."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Reconstruct — choose the session's scene.json", str(Path.home()),
            "Session (scene.json);;JSON files (*.json)",
        )
        if path:
            self.start_reconstruction(Path(path))

    def start_reconstruction(self, path: Path) -> bool:
        """
        Show a saved session's obstacles where they belong and compare them
        with the live scene. Needs a connection (or the mock scene).
        """
        if self._client is None:
            QMessageBox.warning(self, "Not connected",
                                "Connect to Motive first: reconstruction compares live poses.")
            return False
        try:
            session = load_session(path)
        except (OSError, ValueError) as e:
            logger.exception("Could not open session")
            QMessageBox.critical(self, "Could not open session", str(e))
            return False

        self._recon_session = session
        self._recon_targets = targets_from_session(session)
        self._recon_extra_hulls = {}
        self._recon_name.setText(f"Session: {session.session_dir.name}")
        self._placement_table.clear_rows()
        self._viewport.set_selected(None)
        self._viewport.update_scene(
            dict(session.hulls), session.boundary,
            show_boundary = self._chk_show_boundary.isChecked(),
            cameras       = session.boundary.corner_cameras if session.boundary else [],
        )
        self._viewport.update_poses([])          # live poses come from the next update
        self._viewport.set_targets({
            name: (t.position, t.orientation) for name, t in self._recon_targets.items()
        })
        self._live_timer.start()                 # also for the mock scene
        self._update_ui_state()
        self._update_reconstruction()
        self._set_status(f"Reconstructing {session.session_dir.name}: "
                         f"{len(self._recon_targets)} obstacles")
        return True

    @pyqtSlot()
    def _stop_reconstruction(self):
        """Leave reconstruct mode and return to the live scene."""
        self._end_reconstruction()
        self._rebuild_geometry()
        if not isinstance(self._client, NatNetConnection):
            self._live_timer.stop()              # the mock scene has no live updates
        self._update_ui_state()
        self._set_status("Reconstruction stopped.")

    def _end_reconstruction(self):
        self._recon_session = None
        self._recon_targets = {}
        self._placements    = {}
        self._recon_extra_hulls = {}
        self._viewport.set_targets(None)
        self._placement_table.clear_rows()

    def _update_reconstruction(self):
        """Compare the latest live frame with the targets and update the views."""
        if self._recon_session is None or self._client is None:
            return
        latest = self._client.latest_frame
        bodies = convert_frame(latest).rigid_bodies if latest is not None else []
        placements = check_placement(
            self._recon_targets, bodies,
            position_tol = self._recon_pos_tol.value() / 100.0,
            angle_tol    = self._recon_angle_tol.value(),
        )
        self._add_extra_hulls(bodies, placements)
        tracked = [b for b in bodies if b.tracking_valid and (
                   b.name in self._recon_targets or self._recon_extra_hulls.get(b.name))]
        self._viewport.update_poses(tracked)
        self._viewport.set_status_colours(
            {name: STATUS_COLOURS[p.state] for name, p in placements.items()})
        self._placement_table.show_placements(placements)
        self._recon_summary.setText(summary(placements))
        done = is_complete(placements)
        self._recon_summary.setStyleSheet(f"color: {'#27ae60' if done else '#c0392b'};")
        self._placements = placements

    def _add_extra_hulls(self, bodies: list, placements: dict):
        """
        Live obstacles that are not in the session have no saved shape: build
        each one's hull from its live markers, once, and add it to the view.
        """
        live = {b.name: b for b in bodies}
        new = [name for name, p in placements.items()
               if p.state is State.EXTRA and name not in self._recon_extra_hulls]
        if not new:
            return
        for name in new:
            self._recon_extra_hulls[name] = build_hull(live[name])   # None: no hull
        hulls = dict(self._recon_session.hulls)
        hulls.update({n: h for n, h in self._recon_extra_hulls.items() if h is not None})
        self._viewport.update_scene(
            hulls, self._recon_session.boundary,
            show_boundary = self._chk_show_boundary.isChecked(),
        )

    def _current_bodies(self) -> list:
        """Rigid bodies behind the current hulls (live/mock frame or session)."""
        if self._session is not None:
            return self._session.bodies
        return self._world_frame.rigid_bodies if self._world_frame else []

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _start_worker(self, worker: QThread):
        self._worker = worker
        if isinstance(worker, _ConnectWorker):
            worker.connected.connect(self._on_connect_result)
            worker.frame_ready.connect(self._on_frame_ready)
        elif isinstance(worker, _CaptureWorker):
            worker.frame_ready.connect(self._on_frame_ready)
            worker.failed.connect(lambda msg: self._set_status(msg))
        worker.finished.connect(self._on_worker_finished)
        worker.start()
        self._update_ui_state()

    @pyqtSlot()
    def _on_worker_finished(self):
        self._worker = None
        self._update_ui_state()

    @pyqtSlot(bool, str)
    def _on_connect_result(self, success: bool, message: str):
        if success:
            self._connection_label.setText(
                f"Connected — {self._ip_input.text()}"
            )
            self._live_timer.start()
        else:
            self._client = None
            QMessageBox.critical(self, "Connection failed", message)
        self._set_status(message)
        self._update_ui_state()

    @pyqtSlot(object)
    def _on_frame_ready(self, frame: Frame):
        """
        Called when a new Frame arrives (from worker or mock).
        Converts coordinates, rebuilds geometry, updates all widgets.
        """
        self._raw_frame = frame

        # Convert to world frame
        z_up = self._radio_zup.isChecked()
        self._world_frame = convert_frame(frame, z_up=z_up)

        # Rebuild geometry
        self._rebuild_geometry()

        # Count obstacles (bodies that produced a hull), not every streamed
        # body: the "boundary" body and untracked/degenerate bodies are not
        # obstacles.
        n = len(self._hulls)
        status = f"Frame #{frame.frame_number} — {n} obstacle{'' if n == 1 else 's'}"
        if self._skipped_bodies:
            status += f" (skipped: {', '.join(self._skipped_bodies)})"
        self._set_status(status)

    # ------------------------------------------------------------------
    # Geometry pipeline
    # ------------------------------------------------------------------

    def _rebuild_geometry(self):
        """
        Re-run hull and boundary computation from the current world frame.
        Called after a new frame arrives or convention changes.
        """
        if self._world_frame is None:
            return

        # The "boundary" rigid body marks an interest area, never an obstacle
        # (see marker_placement_guide.md).
        boundary_body = next(
            (b for b in self._world_frame.rigid_bodies
             if b.name.lower() == BOUNDARY_BODY_NAME),
            None,
        )
        obstacle_bodies = [
            b for b in self._world_frame.rigid_bodies if b is not boundary_body
        ]

        self._hulls = build_all_hulls(obstacle_bodies)
        # Bodies that could not become obstacles (untracked, too few markers…)
        self._skipped_bodies = [b.name for b in obstacle_bodies if b.name not in self._hulls]

        # Boundary
        wall_height = (
            self._wall_height_spin.value()
            if self._chk_walls.isChecked()
            else None
        )
        wall_in = self._radio_inward.isChecked()

        self._boundary = build_boundary(
            self._world_frame.cameras,
            wall_height_m   = wall_height,
            wall_extrude_in = wall_in,
            mode            = self._boundary_source.currentData(),
            boundary_body   = boundary_body,
        )

        # Update child widgets
        self._obstacle_table.update_from_hulls(
            self._hulls,
            self._world_frame.rigid_bodies,
        )
        self._viewport.update_scene(
            self._hulls,
            self._boundary,
            show_boundary = self._chk_show_boundary.isChecked(),
            bodies        = self._world_frame.rigid_bodies if self._world_frame else None,
            cameras       = self._world_frame.cameras if self._world_frame else None,
        )

        self.geometry_updated.emit()

    # ------------------------------------------------------------------
    # Slots — sidebar controls
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_convention_changed(self, z_up_checked: bool):
        """Re-convert and rebuild when axis convention switches."""
        if self._raw_frame is None:
            return
        self._z_up = self._radio_zup.isChecked()
        self._world_frame = convert_frame(self._raw_frame, z_up=self._z_up)
        self._rebuild_geometry()

    @pyqtSlot(bool)
    def _on_walls_toggled(self, checked: bool):
        self._wall_height_spin.setEnabled(checked)
        self._radio_outward.setEnabled(checked)
        self._radio_inward.setEnabled(checked)
        self._rebuild_geometry()

    @pyqtSlot(bool)
    def _on_boundary_display_changed(self, checked: bool):
        self._viewport.set_boundary_visible(checked)

    # ------------------------------------------------------------------
    # Slots — obstacle table
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_obstacle_selected(self, name: str):
        """Highlight the selected obstacle in the viewport."""
        self._viewport.set_selected(name)

    @pyqtSlot(str, str)
    def _on_obstacle_renamed(self, old_name: str, new_name: str):
        """
        Rename an obstacle — update hulls dict and viewport.
        The rigid body name in the frame is not changed (it comes from
        Motive), but the export name is updated here.
        """
        if old_name not in self._hulls:
            return
        hull = self._hulls.pop(old_name)
        hull.name = new_name
        self._hulls[new_name] = hull

        # Also update the rigid body name so scene.json is consistent
        for body in self._current_bodies():
            if body.name == old_name:
                body.name = new_name
                break

        # Pass the bodies so the viewport re-keys its poses to the new name
        # (otherwise the renamed hull would be drawn at the origin)
        self._viewport.update_scene(
            self._hulls,
            self._boundary,
            show_boundary = self._chk_show_boundary.isChecked(),
            bodies        = self._current_bodies(),
        )
        self._set_status(f"Renamed '{old_name}' → '{new_name}'")

    # ------------------------------------------------------------------
    # Slots — live preview tick
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_live_tick(self):
        """
        Called at 10 Hz while connected. If the latest frame from the
        client has a new frame number, update the viewport pose display.
        This gives a lightweight live pose preview without re-running
        the full hull pipeline on every tick.
        """
        if not self._client or not self._client.is_connected:
            return
        if self._recon_session is not None:
            self._update_reconstruction()
            return
        latest = self._client.latest_frame
        if latest is None:
            return
        # Only update if frame number changed
        if (self._raw_frame is not None and
                latest.frame_number == self._raw_frame.frame_number):
            return
        # Update poses in viewport without rebuilding hulls
        world = convert_frame(latest, z_up=self._radio_zup.isChecked())
        self._viewport.update_poses(world.rigid_bodies)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_export(self):
        if not self._hulls:
            QMessageBox.warning(self, "Nothing to export",
                                "Connect and capture a frame first.")
            return

        # Ask user where to save
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose export directory",
            str(Path.home()),
        )
        if not output_dir:
            return   # user cancelled

        output_path = Path(output_dir)

        try:
            session = build_session(
                hulls           = self._hulls,
                boundary        = self._boundary,
                rigid_bodies    = self._world_frame.rigid_bodies,
                z_up            = self._radio_zup.isChecked(),
                wall_height     = (
                    self._wall_height_spin.value()
                    if self._chk_walls.isChecked() else None
                ),
                wall_extrude_in = self._radio_inward.isChecked(),
                notes           = self._notes_input.text().strip(),
            )
            session_dir = export_session(session, output_path)
            if self._chk_gazebo.isChecked():
                gazebo.write_world(session_dir)
            if self._chk_octomap.isChecked():
                octomap_adapter.write_octomap(session_dir)

            self._export_status.setText(f"Saved to:\n{session_dir.name}")
            self._set_status(f"Exported → {session_dir}")
            QMessageBox.information(
                self,
                "Export complete",
                f"Session exported to:\n{session_dir}",
            )
        except Exception as e:
            logger.exception("Export failed")
            QMessageBox.critical(self, "Export failed", str(e))

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _open_guide(self, relative: str):
        if self._doc_viewer is None:
            self._doc_viewer = DocViewer(self)
        open_guide(self._doc_viewer, relative)

    def _on_about(self):
        self._about_box().exec()

    def _about_box(self) -> QMessageBox:
        box = QMessageBox(self)
        box.setWindowTitle("About OptiTrack Obstacle Builder")
        box.setIconPixmap(QPixmap(str(RESOURCES / "mrs_logo.png")).scaledToWidth(
            280, Qt.TransformationMode.SmoothTransformation))
        box.setText("<b>OptiTrack Obstacle Builder</b><br>MRS — Multi-Robot Systems Lab")
        box.setInformativeText(
            "Generates 3D obstacle models from OptiTrack rigid bodies.\n\n"
            "Outputs: STL files + scene.json\n"
            "Coordinate convention: Z-up (default)\n\n"
            "The guides are in the Help menu."
        )
        return box

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent):
        """Clean up connection on close."""
        self._live_timer.stop()
        if self._client:
            self._client.disconnect()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _avoid_gtk3_wayland_crash(environ=os.environ):
    """
    On GNOME 50 (e.g. Ubuntu 26.04), GTK 3 aborts when it opens a Wayland
    display: it reads a settings key GNOME removed ("antialiasing"). Qt loads
    GTK 3 for the desktop's look, so the app would crash on start. GTK in
    this process uses X11 (XWayland) instead; Qt itself stays on Wayland.
    Only when both are available and the user has not chosen a GTK backend.
    """
    if (sys.platform.startswith("linux") and "GDK_BACKEND" not in environ
            and environ.get("WAYLAND_DISPLAY") and environ.get("DISPLAY")):
        environ["GDK_BACKEND"] = "x11"


def main():
    _avoid_gtk3_wayland_crash()
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("OptiTrack Obstacle Builder")
    app.setStyle("Fusion")   # consistent look on all platforms

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()