"""
Placement table — reconstruction status per obstacle.

Each row:
    ● status colour  |  name  |  move (cm)  |  turn (°)

Numbers only, units in the headers, so it fits the narrow sidebar.

The move cell's tooltip spells out the direction (e.g. "+x 12 cm, -y 3 cm").

Signals:
    selection_changed(name: str) — user clicked a row
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

from ..reconstruct import Placement, State, describe_move

# Status colours (RGB 0-1), shared with the viewport
STATUS_COLOURS = {
    State.PLACED:    (0.18, 0.80, 0.34),   # green
    State.MISPLACED: (0.90, 0.22, 0.20),   # red
    State.MISSING:   (0.55, 0.55, 0.55),   # grey
}


def _qcolor(rgb) -> QColor:
    return QColor.fromRgbF(*rgb)


class PlacementTable(QTableWidget):

    selection_changed = pyqtSignal(str)

    _COLUMNS = ("", "Obstacle", "cm", "°")

    def __init__(self, parent=None):
        super().__init__(0, len(self._COLUMNS), parent)
        self.setHorizontalHeaderLabels(self._COLUMNS)
        header = self.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(0, 20)
        self.setColumnWidth(2, 46)
        self.setColumnWidth(3, 40)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalHeaderItem(2).setToolTip("How far to move it (cm), at its centre")
        self.horizontalHeaderItem(3).setToolTip("How far to turn it (degrees)")
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self._names: list[str] = []
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def show_placements(self, placements: dict[str, Placement]):
        """Show one row per obstacle; rows are kept (and selection too) across updates."""
        names = list(placements)
        if names != self._names:
            self.blockSignals(True)
            self.setRowCount(len(names))
            for row, name in enumerate(names):
                for col in range(len(self._COLUMNS)):
                    self.setItem(row, col, QTableWidgetItem())
                self.item(row, 1).setText(name)
            self._names = names
            self.blockSignals(False)

        for row, name in enumerate(names):
            p = placements[name]
            dot = self.item(row, 0)
            dot.setText("●")
            dot.setForeground(QBrush(_qcolor(STATUS_COLOURS[p.state])))
            dot.setToolTip(p.state.value)
            dot.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            move, turn = self.item(row, 2), self.item(row, 3)
            self.item(row, 1).setToolTip(name)
            if p.state is State.MISSING:
                move.setText("—")
                turn.setText("—")
                for item in (move, turn):
                    item.setToolTip("Not tracked: not streamed by Motive, or not visible now")
            else:
                move.setText(f"{p.distance * 100:.1f}")
                move.setToolTip(f"Move it: {describe_move(p.move)}")
                turn.setText(f"{p.angle_deg:.1f}")
                turn.setToolTip(f"Turn it {p.angle_deg:.1f}°")
            for item in (move, turn):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def clear_rows(self):
        self.setRowCount(0)
        self._names = []

    def _on_selection_changed(self):
        rows = self.selectionModel().selectedRows()
        if rows:
            self.selection_changed.emit(self._names[rows[0].row()])
