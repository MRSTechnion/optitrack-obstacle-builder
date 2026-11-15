"""
3D OpenGL viewport widget.

Renders:
  - World origin axes (X=red, Y=green, Z=blue)
  - Room boundary floor polygon (blue wireframe + transparent fill)
  - Obstacle convex hulls (coloured, semi-transparent faces + wireframe)
  - Selected obstacle highlighted with brighter colour and outline
  - Camera positions as small dots (optional)
  - Reconstruction (set_targets): each obstacle's saved pose as a ghost
    (wireframe), the live obstacle in its status colour, and a line from
    the live obstacle to where it belongs

Navigation:
  - Left-drag   : pan
  - Right-drag  : orbit (rotate around scene centre)
  - Scroll wheel: zoom

The widget is intentionally self-contained — it only knows about
HullResult and BoundaryResult, never about NatNet or the scene writer.
"""

from __future__ import annotations

import math

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_LINE_LOOP,
    GL_LINE_SMOOTH,
    GL_LINE_STIPPLE,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_POLYGON_OFFSET_FILL,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glColorMaterial,
    glDisable,
    glEnable,
    glEnd,
    glLineStipple,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glPolygonMode,
    glPolygonOffset,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from scipy.spatial.transform import Rotation

from ..geometry.hull import BoundaryResult, HullResult

# Obstacle colours — same palette as the table
_COLOURS = [
    (0.216, 0.478, 0.678),   # blue
    (0.847, 0.353, 0.188),   # orange-red
    (0.114, 0.620, 0.459),   # green
    (0.557, 0.267, 0.678),   # purple
    (0.902, 0.494, 0.133),   # amber
    (0.161, 0.502, 0.725),   # steel blue
    (0.753, 0.224, 0.169),   # red
    (0.086, 0.627, 0.522),   # teal
]

_BOUNDARY_COLOUR = (0.302, 0.533, 0.827)   # muted blue
_AXIS_LENGTH     = 0.5                      # metres
_GRID_SIZE       = 3.0                      # half-extent of floor grid
_GRID_STEP       = 0.5                      # grid spacing in metres


def _to_world(point, position, quaternion) -> tuple[float, float, float]:
    """A body-frame point placed at a (position, quaternion [x, y, z, w]) pose."""
    x, y, z = Rotation.from_quat(quaternion).apply(point) + np.asarray(position)
    return float(x), float(y), float(z)


class Viewport3D(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene data
        self._hulls:    dict[str, HullResult]   = {}
        self._boundary: BoundaryResult | None = None
        self._show_boundary = True
        self._selected: str | None = None

        # World-space pose per obstacle: name → (position, quaternion)
        # position  = (x, y, z)
        # quaternion = (qx, qy, qz, qw)
        self._poses: dict[str, tuple] = {}

        # Colour assignment — stable across updates
        self._colour_map: dict[str, tuple] = {}

        # Reconstruction: target pose per obstacle (name → (position, quaternion))
        # and a status colour per obstacle that overrides the palette. Empty =
        # normal view.
        self._targets: dict[str, tuple] = {}
        self._status_colours: dict[str, tuple] = {}

        # All camera positions (world frame) — drawn as small dim dots; the
        # ones at the boundary polygon's vertices are drawn brighter on top.
        self._cameras: list = []

        # Camera / navigation state
        self._azimuth   =  45.0    # degrees, rotation around Z
        self._elevation =  30.0    # degrees, tilt up from horizontal
        self._distance  =   6.0    # metres from scene centre
        self._pan_x     =   0.0    # pan in world X
        self._pan_y     =   0.0    # pan in world Y

        self._last_mouse: QPoint | None = None
        self._mouse_btn:  Qt.MouseButton | None = None

        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def update_scene(
        self,
        hulls:         dict[str, HullResult],
        boundary:      BoundaryResult | None,
        show_boundary: bool = True,
        bodies:        list | None = None,   # list[RigidBody] — for world poses
        cameras:       list | None = None,   # list[CameraInfo] — all cameras
    ):
        """Replace the current scene data and redraw."""
        self._hulls         = hulls
        self._boundary      = boundary
        self._show_boundary = show_boundary

        if cameras is not None:
            self._cameras = list(cameras)

        # Store world-space poses from rigid bodies
        if bodies:
            self._poses = {
                b.name: (b.position.as_tuple(), b.orientation.as_tuple())
                for b in bodies
            }

        # Assign colours — preserve existing assignments on rename
        assigned  = set(self._colour_map.keys())
        new_names = set(hulls.keys())

        for gone in assigned - new_names:
            del self._colour_map[gone]

        idx = len(self._colour_map)
        for name in hulls:
            if name not in self._colour_map:
                self._colour_map[name] = _COLOURS[idx % len(_COLOURS)]
                idx += 1

        self.update()

    def update_poses(self, bodies):
        """
        Lightweight pose update — refreshes world-space poses without
        rebuilding hull geometry. Used by the live tick at 10 Hz.
        """
        self._poses = {
            b.name: (b.position.as_tuple(), b.orientation.as_tuple())
            for b in bodies
        }
        self.update()

    def set_targets(self, targets: dict[str, tuple] | None):
        """
        Show where each obstacle belongs (name → (position, quaternion)), or
        None to leave reconstruction. While targets are set, obstacles with
        no live pose (not tracked) are drawn only as their target ghost, and
        hulls that are not targets (extra live obstacles) only when tracked.
        """
        self._targets = dict(targets or {})
        if not self._targets:
            self._status_colours = {}
        self.update()

    def set_status_colours(self, colours: dict[str, tuple]):
        """Per-obstacle RGB colours (0-1) that replace the palette colour."""
        self._status_colours = dict(colours)
        self.update()

    def set_selected(self, name: str | None):
        """Highlight the named obstacle (or clear selection if None)."""
        self._selected = name
        self.update()

    def set_boundary_visible(self, visible: bool):
        self._show_boundary = visible
        self.update()

    def clear(self):
        self._hulls.clear()
        self._boundary    = None
        self._selected    = None
        self._colour_map.clear()
        self._poses.clear()
        self._targets.clear()
        self._status_colours.clear()
        self.update()

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        glClearColor(0.13, 0.14, 0.16, 1.0)   # dark background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply camera transform
        glTranslatef(0.0, 0.0, -self._distance)
        glRotatef(-self._elevation, 1.0, 0.0, 0.0)
        glRotatef( self._azimuth,   0.0, 0.0, 1.0)
        glTranslatef(-self._pan_x, -self._pan_y, 0.0)

        # Scene elements
        self._draw_grid()
        self._draw_axes()
        self._draw_cameras()

        if self._show_boundary and self._boundary is not None:
            self._draw_boundary(self._boundary)

        for name, hull in self._hulls.items():
            colour    = self._status_colours.get(name) or self._colour_map.get(name, _COLOURS[0])
            selected  = (name == self._selected)
            pose      = self._poses.get(name)
            if name in self._targets:
                self._draw_ghost(hull, colour, self._targets[name])
            if self._targets and pose is None:
                continue                      # reconstructing, not tracked: no live shape
            self._draw_hull(hull, colour, selected)

        # Guide lines last, over everything
        for name, hull in self._hulls.items():
            if name in self._targets and name in self._poses:
                colour = self._status_colours.get(name) or self._colour_map.get(name, _COLOURS[0])
                self._draw_guide(hull, colour, self._poses[name], self._targets[name])

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self):
        """Draw a subtle floor grid at Z=0."""
        glLineWidth(0.5)
        glColor4f(0.3, 0.3, 0.3, 0.6)
        glBegin(GL_LINES)
        steps = int(_GRID_SIZE / _GRID_STEP)
        for i in range(-steps, steps + 1):
            x = i * _GRID_STEP
            glVertex3f(x,        -_GRID_SIZE, 0.0)
            glVertex3f(x,         _GRID_SIZE, 0.0)
            glVertex3f(-_GRID_SIZE, x,        0.0)
            glVertex3f( _GRID_SIZE, x,        0.0)
        glEnd()

    def _draw_axes(self):
        """Draw X (red), Y (green), Z (blue) axes at the world origin."""
        glLineWidth(2.0)
        glBegin(GL_LINES)

        glColor4f(0.9, 0.2, 0.2, 1.0)        # X — red
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(_AXIS_LENGTH, 0.0, 0.0)

        glColor4f(0.2, 0.8, 0.2, 1.0)        # Y — green
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, _AXIS_LENGTH, 0.0)

        glColor4f(0.2, 0.4, 0.9, 1.0)        # Z — blue
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, _AXIS_LENGTH)

        glEnd()

        # Small axis labels as coloured dots at the tips
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor4f(0.9, 0.2, 0.2, 1.0); glVertex3f(_AXIS_LENGTH, 0.0, 0.0)
        glColor4f(0.2, 0.8, 0.2, 1.0); glVertex3f(0.0, _AXIS_LENGTH, 0.0)
        glColor4f(0.2, 0.4, 0.9, 1.0); glVertex3f(0.0, 0.0, _AXIS_LENGTH)
        glEnd()

    def _draw_cameras(self):
        """All cameras as small dim dots at their real positions."""
        if not self._cameras:
            return
        r, g, b = _BOUNDARY_COLOUR
        glPointSize(4.0)
        glColor4f(r, g, b, 0.45)
        glBegin(GL_POINTS)
        for cam in self._cameras:
            glVertex3f(cam.position.x, cam.position.y, cam.position.z)
        glEnd()

    def _draw_boundary(self, boundary: BoundaryResult):
        """Draw the room boundary floor polygon and optional walls."""
        r, g, b = _BOUNDARY_COLOUR
        verts = boundary.floor_vertices
        faces = boundary.floor_faces

        # Filled floor — very transparent
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glColor4f(r, g, b, 0.08)
        self._draw_faces(verts, faces)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Floor outline
        glLineWidth(1.5)
        glColor4f(r, g, b, 0.7)
        glBegin(GL_LINE_LOOP)
        for pt in verts:
            glVertex3f(pt[0], pt[1], pt[2])
        glEnd()

        # Corner camera positions
        glPointSize(6.0)
        glColor4f(r, g, b, 1.0)
        glBegin(GL_POINTS)
        for cam in boundary.corner_cameras:
            glVertex3f(cam.position.x, cam.position.y, cam.position.z)
        glEnd()

        # Wall panels
        for verts_w, faces_w in boundary.wall_meshes:
            glColor4f(r, g, b, 0.12)
            self._draw_faces(verts_w, faces_w)
            glLineWidth(0.8)
            glColor4f(r, g, b, 0.4)
            self._draw_wireframe(verts_w, faces_w)

    def _draw_hull(
        self,
        hull:     HullResult,
        colour:   tuple,
        selected: bool,
    ):
        """
        Draw one obstacle hull — filled faces + wireframe overlay.

        Hull vertices are in body-local frame (centred at origin).
        We apply the body's world-space pose as a GL matrix transform
        so each obstacle appears at its correct position and orientation.
        """
        r, g, b = colour
        verts   = hull.vertices
        faces   = hull.faces

        glPushMatrix()
        self._apply_body_pose(hull.name)

        # Filled faces — semi-transparent
        fill_alpha = 0.75 if selected else 0.45
        glColor4f(r, g, b, fill_alpha)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self._draw_faces(verts, faces)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Wireframe overlay
        wire_alpha = 1.0 if selected else 0.7
        wire_width = 2.5 if selected else 1.0
        glLineWidth(wire_width)
        glColor4f(
            min(r * 1.3, 1.0),
            min(g * 1.3, 1.0),
            min(b * 1.3, 1.0),
            wire_alpha,
        )
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self._draw_faces(verts, faces)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Selection outline — bright white halo
        if selected:
            glLineWidth(3.5)
            glColor4f(1.0, 1.0, 1.0, 0.9)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            self._draw_faces(verts, faces)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()

    def _draw_ghost(self, hull: HullResult, colour: tuple, pose: tuple):
        """The obstacle at its target pose: a dashed outline, no fill."""
        r, g, b = colour
        glPushMatrix()
        self._apply_pose(pose)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0x0F0F)                  # dashes
        glLineWidth(2.0)
        glColor4f(min(r * 1.3, 1.0), min(g * 1.3, 1.0), min(b * 1.3, 1.0), 1.0)
        self._draw_wireframe(hull.vertices, hull.faces)
        glDisable(GL_LINE_STIPPLE)
        glPopMatrix()

    def _draw_guide(self, hull: HullResult, colour: tuple, live: tuple, target: tuple):
        """A line from the live obstacle's centre to its target's centre."""
        centre = np.asarray(hull.vertices, dtype=float).mean(axis=0)
        a, b = (_to_world(centre, *live), _to_world(centre, *target))
        glDisable(GL_DEPTH_TEST)                  # visible through the obstacles
        glLineWidth(2.5)
        glColor4f(1.0, 1.0, 1.0, 0.9)
        glBegin(GL_LINES)
        glVertex3f(*a)
        glVertex3f(*b)
        glEnd()
        glPointSize(9.0)
        glColor4f(*colour, 1.0)
        glBegin(GL_POINTS)
        glVertex3f(*b)                            # where its centre belongs
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def _apply_body_pose(self, name: str):
        """
        Apply the world-space pose of the named body as a GL transform.

        Converts the stored quaternion to an axis-angle representation
        for glRotatef, then applies the translation.

        If no pose is stored for this name (e.g. during mock/fixture use
        before poses are populated) the identity transform is applied and
        the hull renders at the world origin.
        """
        pose = self._poses.get(name)
        if pose is not None:
            self._apply_pose(pose)

    @staticmethod
    def _apply_pose(pose: tuple):
        """Apply a (position, quaternion) pose as a GL transform."""
        position, quaternion = pose
        px, py, pz           = position
        qx, qy, qz, qw       = quaternion

        # Translate to world position
        glTranslatef(px, py, pz)

        # Convert quaternion to axis-angle for glRotatef
        # Clamp qw to [-1, 1] to avoid acos domain errors from float noise
        qw_clamped = max(-1.0, min(1.0, qw))
        angle_rad  = 2.0 * math.acos(qw_clamped)
        angle_deg  = math.degrees(angle_rad)

        sin_half = math.sqrt(max(0.0, 1.0 - qw_clamped * qw_clamped))

        if sin_half < 1e-6:
            # Rotation is negligible — use Z axis as default (no rotation)
            ax, ay, az = 0.0, 0.0, 1.0
        else:
            ax = qx / sin_half
            ay = qy / sin_half
            az = qz / sin_half

        glRotatef(angle_deg, ax, ay, az)

    @staticmethod
    def _draw_faces(vertices: np.ndarray, faces: np.ndarray):
        """Draw filled triangles."""
        glBegin(GL_TRIANGLES)
        for face in faces:
            for idx in face:
                v = vertices[idx]
                glVertex3f(float(v[0]), float(v[1]), float(v[2]))
        glEnd()

    @staticmethod
    def _draw_wireframe(vertices: np.ndarray, faces: np.ndarray):
        """Draw wireframe edges."""
        glBegin(GL_LINES)
        for face in faces:
            for i in range(3):
                v0 = vertices[face[i]]
                v1 = vertices[face[(i + 1) % 3]]
                glVertex3f(float(v0[0]), float(v0[1]), float(v0[2]))
                glVertex3f(float(v1[0]), float(v1[1]), float(v1[2]))
        glEnd()

    # ------------------------------------------------------------------
    # Mouse navigation
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.pos()
        self._mouse_btn  = event.button()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            return

        dx = event.pos().x() - self._last_mouse.x()
        dy = event.pos().y() - self._last_mouse.y()
        self._last_mouse = event.pos()

        if self._mouse_btn == Qt.MouseButton.RightButton:
            # Orbit
            self._azimuth   += dx * 0.5
            self._elevation  = max(-89.0, min(89.0, self._elevation + dy * 0.5))

        elif self._mouse_btn == Qt.MouseButton.LeftButton:
            # Pan — scale by distance so panning feels consistent at any zoom
            pan_scale = self._distance * 0.002
            self._pan_x -= dx * pan_scale
            self._pan_y += dy * pan_scale

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_mouse = None
        self._mouse_btn  = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        # Zoom in/out — clamp to reasonable range
        self._distance *= 0.999 ** delta
        self._distance  = max(0.5, min(50.0, self._distance))
        self.update()

    def keyPressEvent(self, event):
        """Reset view on R key."""
        if event.key() == Qt.Key.Key_R:
            self._azimuth   =  45.0
            self._elevation =  30.0
            self._distance  =   6.0
            self._pan_x     =   0.0
            self._pan_y     =   0.0
            self.update()
        else:
            super().keyPressEvent(event)