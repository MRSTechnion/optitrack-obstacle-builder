"""
Obstacle table widget — shows rigid bodies with inline rename support.

Each row:
    ● colour dot  |  name (editable)  |  marker count  |  volume

Signals:
    selection_changed(name: str)        — user clicked a row
    name_changed(old_name, new_name)    — user committed a rename
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from ..geometry.hull import HullResult
from ..natnet.types import RigidBody

# Obstacle colours — cycled through as bodies are added
_COLOURS = [
    "#378ADD", "#D85A30", "#1D9E75", "#8E44AD",
    "#E67E22", "#2980B9", "#C0392B", "#16A085",
]


class ObstacleTable(QTableWidget):

    selection_changed = pyqtSignal(str)        # body name
    name_changed      = pyqtSignal(str, str)   # old, new

    _COLUMNS = ["", "Name", "Markers", "Vol (cm³)"]

    def __init__(self, parent=None):
        super().__init__(0, len(self._COLUMNS), parent)

        self.setHorizontalHeaderLabels(self._COLUMNS)
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self.setColumnWidth(0, 20)
        self.setColumnWidth(2, 60)
        self.setColumnWidth(3, 70)

        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

        # Internal state
        self._name_to_row:  dict[str, int]   = {}
        self._row_to_name:  dict[int, str]   = {}
        self._row_colours:  dict[int, str]   = {}
        self._editing_name: dict[int, str]   = {}   # row → original name

        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_hulls(
        self,
        hulls:   dict[str, HullResult],
        bodies:  list[RigidBody],
    ):
        """
        Rebuild the table from the current hull results.
        Preserves any renames the user has already made.
        """
        # Block signals while rebuilding to avoid spurious itemChanged events
        self.blockSignals(True)
        self.setRowCount(0)
        self._name_to_row.clear()
        self._row_to_name.clear()
        self._row_colours.clear()

        body_map = {b.name: b for b in bodies}

        for i, (name, hull) in enumerate(hulls.items()):
            colour = _COLOURS[i % len(_COLOURS)]
            body   = body_map.get(name)
            markers = body.marker_count if body else len(hull.vertices)
            volume_cm3 = round(hull.volume * 1e6, 1)

            self.insertRow(i)
            self._add_row(i, name, markers, volume_cm3, colour)

        self.blockSignals(False)

    def clear(self):
        self.setRowCount(0)
        self._name_to_row.clear()
        self._row_to_name.clear()
        self._row_colours.clear()

    def colour_for(self, name: str) -> str:
        """Return the hex colour for the named obstacle."""
        row = self._name_to_row.get(name)
        if row is None:
            return "#888888"
        return self._row_colours.get(row, "#888888")

    def select_by_name(self, name: str):
        """Select the row with the given obstacle name."""
        row = self._name_to_row.get(name)
        if row is not None:
            self.selectRow(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_row(
        self,
        row:        int,
        name:       str,
        markers:    int,
        volume_cm3: float,
        colour:     str,
    ):
        # Colour dot cell (not editable)
        dot = QTableWidgetItem()
        dot.setBackground(QBrush(QColor(colour)))
        dot.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.setItem(row, 0, dot)

        # Name cell (editable on double-click)
        name_item = QTableWidgetItem(name)
        name_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled |
            Qt.ItemFlag.ItemIsSelectable |
            Qt.ItemFlag.ItemIsEditable
        )
        self.setItem(row, 1, name_item)

        # Marker count (not editable)
        mk = QTableWidgetItem(str(markers))
        mk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        mk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, 2, mk)

        # Volume (not editable)
        vol = QTableWidgetItem(str(volume_cm3))
        vol.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        vol.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, 3, vol)

        self._name_to_row[name] = row
        self._row_to_name[row]  = name
        self._row_colours[row]  = colour

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        rows = self.selectionModel().selectedRows()
        if rows:
            row  = rows[0].row()
            name = self._row_to_name.get(row, "")
            if name:
                self.selection_changed.emit(name)

    def _on_item_changed(self, item: QTableWidgetItem):
        """Called when any cell is edited. We only care about column 1 (name)."""
        if item.column() != 1:
            return

        row      = item.row()
        new_name = item.text().strip()
        old_name = self._row_to_name.get(row, "")

        if not new_name:
            # Reject empty name — restore original
            self.blockSignals(True)
            item.setText(old_name)
            self.blockSignals(False)
            return

        if new_name == old_name:
            return

        # Check for duplicate names
        if new_name in self._name_to_row:
            self.blockSignals(True)
            item.setText(old_name)
            self.blockSignals(False)
            return

        # Commit rename
        del self._name_to_row[old_name]
        self._name_to_row[new_name] = row
        self._row_to_name[row]      = new_name

        self.name_changed.emit(old_name, new_name)