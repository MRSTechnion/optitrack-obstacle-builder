"""
Setup reconstruction — compare a saved session with the live scene.

To rebuild an experiment setup, each obstacle is put back where it was when
the session was exported. Obstacles are matched to live rigid bodies by
name, and each one is:

    placed     — within the position and angle tolerances of its saved pose
    misplaced  — tracked, but further than a tolerance from it
    missing    — not streamed, or not tracked, right now
    extra      — tracked live, but not in the session (e.g. an obstacle
                 added since): it should not be there

Distances are measured at the centre of the obstacle's hull, not at the
rigid body's pivot, so "move 12 cm" is about the object itself. The saved
hull is used for both poses: the obstacle's rigid body in Motive must be
the same one as at export (same markers, same pivot) — e.g. the same Motive
project, or the rigid body re-created from the same assets.

No Qt here: the GUI and scripts use the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry.hull import BOUNDARY_BODY_NAME
from .natnet.types import RigidBody
from .scene.scene_writer import LoadedSession

POSITION_TOL_M = 0.02     # default: 2 cm
ANGLE_TOL_DEG  = 2.0      # default: 2°


class State(str, Enum):
    PLACED    = "placed"
    MISPLACED = "misplaced"
    MISSING   = "missing"
    EXTRA     = "not in session"


@dataclass(frozen=True)
class Target:
    """Where an obstacle belongs: its saved pose and the hull centre (body frame)."""
    name:        str
    position:    tuple[float, float, float]
    orientation: tuple[float, float, float, float]      # qx, qy, qz, qw
    centre:      np.ndarray                              # hull centre, body frame

    def centre_world(self, position=None, orientation=None) -> np.ndarray:
        """The hull centre in the world, at the saved pose or the given one."""
        p = self.position if position is None else position
        q = self.orientation if orientation is None else orientation
        return Rotation.from_quat(q).apply(self.centre) + np.asarray(p)


@dataclass(frozen=True)
class Placement:
    """How far an obstacle is from its saved pose."""
    name:      str
    state:     State
    move:      np.ndarray | None = None   # world-frame vector to its target (m)
    angle_deg: float | None = None        # rotation still needed (degrees)

    @property
    def distance(self) -> float | None:
        return None if self.move is None else float(np.linalg.norm(self.move))


def targets_from_session(session: LoadedSession) -> dict[str, Target]:
    """One Target per obstacle of a saved session."""
    return {
        body.name: Target(
            name        = body.name,
            position    = body.position.as_tuple(),
            orientation = body.orientation.as_tuple(),
            centre      = np.asarray(session.hulls[body.name].vertices, dtype=float).mean(axis=0),
        )
        for body in session.bodies
    }


def check_placement(
    targets:      dict[str, Target],
    bodies:       list[RigidBody],
    position_tol: float = POSITION_TOL_M,
    angle_tol:    float = ANGLE_TOL_DEG,
) -> dict[str, Placement]:
    """
    Compare live rigid bodies (world frame, Z-up) with the targets.
    Returns a Placement for every target, in the targets' order, then one
    (EXTRA) for every tracked live body that is not a target. The boundary
    rigid body is never an obstacle and is left out.
    """
    live = {b.name: b for b in bodies}
    out = {}
    for name, t in targets.items():
        body = live.get(name)
        if body is None or not body.tracking_valid:
            out[name] = Placement(name, State.MISSING)
            continue
        pos, quat = body.position.as_tuple(), body.orientation.as_tuple()
        move = t.centre_world() - t.centre_world(pos, quat)
        turn = Rotation.from_quat(quat).inv() * Rotation.from_quat(t.orientation)
        angle = float(np.degrees(turn.magnitude()))
        ok = np.linalg.norm(move) <= position_tol and angle <= angle_tol
        out[name] = Placement(name, State.PLACED if ok else State.MISPLACED, move, angle)
    for b in bodies:
        if (b.name not in targets and b.tracking_valid
                and b.name.lower() != BOUNDARY_BODY_NAME):
            out[b.name] = Placement(b.name, State.EXTRA)
    return out


def is_complete(placements: dict[str, Placement]) -> bool:
    """True when every target is in place and nothing extra is in the scene."""
    return all(p.state is State.PLACED for p in placements.values())


def summary(placements: dict[str, Placement]) -> str:
    """e.g. "2 / 3 in place" (or "All 3 in place"), then e.g. "· 1 extra"."""
    states = [p.state for p in placements.values()]
    extra = states.count(State.EXTRA)
    n = len(states) - extra
    k = states.count(State.PLACED)
    text = f"All {n} in place" if n and k == n else f"{k} / {n} in place"
    return f"{text} · {extra} extra" if extra else text


def describe_move(move: np.ndarray) -> str:
    """The move as short directions, largest first, e.g. "+x 12 cm, -y 3 cm"."""
    parts = [(abs(v), f"{'+' if v > 0 else '-'}{axis} {abs(v) * 100:.0f} cm")
             for axis, v in zip("xyz", move, strict=True) if abs(v) >= 0.005]
    return ", ".join(text for _, text in sorted(parts, reverse=True)) or "—"
