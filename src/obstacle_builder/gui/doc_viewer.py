"""
Documentation viewer — the README and docs/ guides, inside the app.

The guides are Markdown files in the repository. They are shown offline,
so they match the installed version and work on a lab network without
internet. Links between guides and images open in the same window; web
links open in the browser.

When the files are not there (an install without the repository), the
guide's page on GitHub is opened instead.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QTextDocument
from PyQt6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout, QWidget

REPO_ROOT = Path(__file__).resolve().parents[3]          # src/obstacle_builder/gui → repo
REPO_URL  = "https://github.com/MRSTechnion/optitrack-obstacle-builder/blob/main/"

# Help menu entries: (label, path from the repository root)
GUIDES = [
    ("User guide",             "README.md"),
    ("Marker placement guide", "docs/marker_placement_guide.md"),
    ("scene.json format",      "docs/scene_json_spec.md"),
    ("Collision checks",       "docs/collision.md"),
]


class _Browser(QTextBrowser):
    """A QTextBrowser that scales wide images (screenshots) down to fit."""

    def loadResource(self, kind: int, url: QUrl):
        data = super().loadResource(kind, url)
        if kind not in (QTextDocument.ResourceType.ImageResource,
                        QTextDocument.ResourceType.ImageResource.value):
            return data
        image = data if isinstance(data, QImage) else QImage.fromData(data)   # file bytes
        width = self.viewport().width() - 40
        if image.isNull() or not image.width() > width > 0:
            return data
        return image.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


class DocViewer(QDialog):
    """A non-modal window showing one Markdown guide at a time."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.resize(820, 760)
        self._browser = _Browser(self)
        self._browser.document().setDocumentMargin(16)
        self._browser.setOpenLinks(False)                 # handled in _on_link
        self._browser.anchorClicked.connect(self._on_link)
        self._browser.sourceChanged.connect(self._on_source_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._browser)

    def show_file(self, path: Path):
        self._browser.setSource(QUrl.fromLocalFile(str(path)),
                                QTextDocument.ResourceType.MarkdownResource)
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_link(self, url: QUrl):
        target = self._browser.source().resolved(url)
        if target.isLocalFile() and target.toLocalFile().endswith(".md"):
            self.show_file(Path(target.toLocalFile()))    # another guide
        elif target.isLocalFile() and Path(target.toLocalFile()).exists():
            QDesktopServices.openUrl(target)              # e.g. an example script
        elif not target.isLocalFile():
            QDesktopServices.openUrl(target)              # web link → browser

    def _on_source_changed(self, url: QUrl):
        title = self._browser.documentTitle() or Path(url.toLocalFile()).stem
        self.setWindowTitle(title)


def open_guide(viewer: DocViewer, relative: str) -> bool:
    """Show a guide in the viewer; open it on GitHub if it is not installed."""
    path = REPO_ROOT / relative
    if path.is_file():
        viewer.show_file(path)
        return True
    QDesktopServices.openUrl(QUrl(REPO_URL + relative))
    return False
