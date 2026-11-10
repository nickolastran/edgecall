"""ABS zone geometry.

Coordinates (Statcast): x = horizontal at the plate, catcher's view (+x = 1B side), z = height,
y = distance from the back tip of home plate (front edge at y = 17 in). Feet throughout,
except signed distances, which are reported in INCHES.

The ABS zone is a 2-D rectangle in the plane at the middle of the plate. A pitch is a
strike if any part of the ball touches it, i.e. the ball CENTER lies in the rectangle
dilated by the ball radius (a rounded rectangle). The signed distance is negative inside
that region and positive outside.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from absengine.config import ZoneRules

PLATE_FRONT_Y_FT = 17.0 / 12.0
Y0_FT = 50.0  # Statcast reference plane for vx0/vy0/vz0


def zone_bounds(height_in: np.ndarray, z: ZoneRules) -> tuple[np.ndarray, np.ndarray]:
    """(top_ft, bot_ft) from certified height."""
    h = np.asarray(height_in, dtype=float) / 12.0
    return h * z.top_frac_height, h * z.bot_frac_height


def project_to_plane(
    plate_x: np.ndarray,
    plate_z: np.ndarray,
    vx0: np.ndarray,
    vy0: np.ndarray,
    vz0: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    depth_in: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the plate crossing from the front edge (y = 17 in) to `depth_in` behind it,
    with the constant-acceleration Statcast trajectory. No release point is needed: the
    velocity at the front edge follows from (vy0, ay) and the distance travelled from y0."""
    y_front = PLATE_FRONT_Y_FT
    vy_f = -np.sqrt(np.maximum(vy0**2 + 2 * ay * (y_front - Y0_FT), 1e-9))  # toward plate: < 0
    t_f = (vy_f - vy0) / ay
    vx_f, vz_f = vx0 + ax * t_f, vz0 + az * t_f
    dy = -depth_in / 12.0  # y decreases as the ball moves toward the catcher
    # solve 0.5*ay*dt^2 + vy_f*dt - dy = 0 for the small positive root
    disc = np.sqrt(np.maximum(vy_f**2 + 2 * ay * dy, 0.0))
    dt = np.where(np.abs(ay) > 1e-9, (-vy_f - disc) / ay, dy / vy_f)
    dt = np.where(dt < 0, (-vy_f + disc) / ay, dt)
    return plate_x + vx_f * dt + 0.5 * ax * dt**2, plate_z + vz_f * dt + 0.5 * az * dt**2


def signed_distance_in(
    x_ft: np.ndarray,
    z_ft: np.ndarray,
    top_ft: np.ndarray,
    bot_ft: np.ndarray,
    zr: ZoneRules,
) -> np.ndarray:
    """Signed distance (inches) from ball center to the strike region
    (rectangle dilated by the ball radius). < 0 = strike under ABS."""
    half_w = zr.plate_width_in / 24.0
    cz = (top_ft + bot_ft) / 2
    half_h = (top_ft - bot_ft) / 2
    qx = np.abs(x_ft) - half_w
    qz = np.abs(z_ft - cz) - half_h
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0))
    inside = np.minimum(np.maximum(qx, qz), 0)
    return np.asarray((outside + inside) * 12.0 - zr.ball_radius_in, dtype=float)


def nearest_edge(
    x_ft: np.ndarray,
    z_ft: np.ndarray,
    top_ft: np.ndarray,
    bot_ft: np.ndarray,
    stand: np.ndarray,
    zr: ZoneRules,
) -> np.ndarray:
    """Which edge decides the call: top | bottom | inside | outside (batter-relative).
    For a pitch outside the rectangle, the axis with the larger violation; inside, the
    closest edge. RHB stand on the -x side (catcher's view), so inside = sign(x) < 0."""
    half_w = zr.plate_width_in / 24.0
    cz = (top_ft + bot_ft) / 2
    qx = np.abs(x_ft) - half_w
    qz = np.abs(z_ft - cz) - (top_ft - bot_ft) / 2
    vertical = qz > qx
    vert = np.where(z_ft >= cz, "top", "bottom")
    inside_side = np.where(np.asarray(stand) == "R", x_ft < 0, x_ft > 0)
    horiz = np.where(inside_side, "inside", "outside")
    return np.where(vertical, vert, horiz)


def add_zone_features(
    df: pl.DataFrame,
    zr: ZoneRules,
    *,
    project: np.ndarray | bool,
    top_col: str = "abs_top_ft",
    bot_col: str = "abs_bot_ft",
    x_col: str = "plate_x",
    z_col: str = "plate_z",
) -> pl.DataFrame:
    """Adds zx_ft, zz_ft (location in the ABS plane), edge_dist_in, edge.

    `project` (per row or scalar): True where the coordinates are at the FRONT edge of the
    plate and must be moved to the mid-plate ABS plane."""
    x, z = df[x_col].to_numpy(), df[z_col].to_numpy()
    mask = np.broadcast_to(np.asarray(project, dtype=bool), x.shape)
    if mask.any():
        v = {c: df[c].to_numpy() for c in ("vx0", "vy0", "vz0", "ax", "ay", "az")}
        px, pz = project_to_plane(x, z, **v, depth_in=zr.plane_depth_in)
        x, z = np.where(mask, px, x), np.where(mask, pz, z)
    top, bot = df[top_col].to_numpy(), df[bot_col].to_numpy()
    return df.with_columns(
        zx_ft=pl.Series(x),
        zz_ft=pl.Series(z),
        edge_dist_in=pl.Series(signed_distance_in(x, z, top, bot, zr)),
        edge=pl.Series(nearest_edge(x, z, top, bot, df["stand"].to_numpy(), zr)),
    )
