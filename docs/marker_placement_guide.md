# Marker placement guide

This guide defines how to attach reflective markers to physical obstacles
so the tool can reconstruct their 3D geometry automatically.

## Core principle

The tool computes a convex hull from the marker positions. The hull
becomes the obstacle's collision and visual geometry. For this to work,
markers must cover the outer corners and edges of the obstacle — not
the flat faces.

## Rules

### Minimum markers per shape

| Shape | Min markers | Placement |
|---|---|---|
| Box / wall | 8 | One per corner |
| Thin wall (flat) | 4 | One per corner (produces flat hull — add height manually if needed) |
| Cylinder / pillar | 6 | 3 around top ring + 3 around base ring |
| Ramp / wedge | 6 | Corners of both triangular faces |
| Irregular shape | 6+ | All outer corners, at least 3 non-collinear per face |

### Placement rules

1. **Corners only** — place markers at vertices, not on flat surfaces.
   A marker in the middle of a face does not add geometry information.

2. **No collinear markers on a single edge** — three markers in a line
   contribute no volume to the hull. Use corners.

3. **Minimum 4 non-coplanar markers** — if all markers are on one plane,
   the hull has zero volume. Obstacles need markers on at least two faces.

4. **Secure the markers firmly** — any movement between capture and export
   will distort the reconstructed shape. Use adhesive mounts or clips.

5. **Avoid marker occlusion** — cameras must see at least 3 markers at all
   times. Place markers on the sides facing the camera ring, not on the
   bottom face of the obstacle.

## Defining the rigid body in Motive

1. Place the obstacle in the capture area with markers attached.
2. In Motive, select all markers belonging to one obstacle.
3. Right-click → Create Rigid Body.
4. Name the rigid body clearly (e.g. `wall_left`, `pillar_1`).
   This name becomes the STL filename and the key in `scene.json`.
5. Repeat for each obstacle.

## Interest area markers (optional)

To limit the boundary to part of the room (an "interest area") instead of
the whole camera volume:

1. Place markers **on the floor** at the corners of the area — 4 for a
   rectangle, more for any other convex shape. Markers inside the area are
   ignored; only the outline counts.
2. In Motive, select them and create a rigid body named exactly `boundary`
   (case-insensitive), and make sure it is published.
3. In the tool, set *Room boundary → Source* to *Interest area*, and press
   *Refresh* if the tool was already connected.

The `boundary` rigid body is never exported as an obstacle, whichever source
is selected. Its markers must be visible at capture time (at least 3, not in
a line).