from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from absengine.config import abs_rules
from absengine.features import zone as Z
from absengine.overturn.labels import wrong_call_dist

ZR = abs_rules().zone
HALF_W = ZR.plate_width_in / 24
R_IN = ZR.ball_radius_in
TOP, BOT = 3.4, 1.6


def d(x: float, z: float) -> float:
    return float(
        Z.signed_distance_in(np.array([x]), np.array([z]), np.array([TOP]), np.array([BOT]), ZR)[0]
    )


def test_rules_from_config() -> None:
    top, bot = Z.zone_bounds(np.array([72.0]), ZR)
    assert top[0] == pytest.approx(72 * 0.535 / 12) and bot[0] == pytest.approx(72 * 0.27 / 12)


def test_ball_radius_boundary() -> None:
    # center exactly one ball radius outside the edge -> touching -> boundary (d = 0)
    assert d(HALF_W + R_IN / 12, 2.5) == pytest.approx(0.0, abs=1e-9)
    assert d(0.0, TOP + R_IN / 12) == pytest.approx(0.0, abs=1e-9)
    assert d(0.0, BOT - R_IN / 12 - 1 / 12) == pytest.approx(1.0)


def test_center_and_corner() -> None:
    assert d(0.0, (TOP + BOT) / 2) == pytest.approx(-HALF_W * 12 - R_IN)  # width is the binding dim
    a, b = 0.1, 0.2
    assert d(HALF_W + a, TOP + b) == pytest.approx(np.hypot(a, b) * 12 - R_IN)


@given(st.floats(-2, 2), st.floats(0, 5))
def test_symmetric_in_x(x: float, z: float) -> None:
    assert d(x, z) == pytest.approx(d(-x, z))


@settings(max_examples=50)
@given(st.floats(0, 2), st.floats(0.001, 1), st.floats(0.5, 4.5))
def test_monotone_outward(x: float, dx: float, z: float) -> None:
    assert d(x + dx, z) >= d(x, z) - 1e-9


@given(st.floats(-2, 2), st.floats(0, 5))
def test_one_lipschitz(x: float, z: float) -> None:
    # signed distance moves at most 12 in per foot of displacement
    assert abs(d(x + 0.01, z) - d(x, z)) <= 0.12 + 1e-9


def test_edges() -> None:
    xs = np.array([0.0, 0.0, -1.2, 1.2, -1.2, 1.2])
    zs = np.array([4.0, 1.0, 2.5, 2.5, 2.5, 2.5])
    stand = np.array(["R", "R", "R", "R", "L", "L"])
    e = Z.nearest_edge(xs, zs, np.full(6, TOP), np.full(6, BOT), stand, ZR)
    assert e.tolist() == ["top", "bottom", "inside", "outside", "outside", "inside"]


def _simulate(
    x0: float,
    z0: float,
    vx0: float,
    vy0: float,
    vz0: float,
    ax: float,
    ay: float,
    az: float,
    y: float,
) -> tuple[float, float]:
    """Brute-force: find t with y(t) = y by bisection, return (x, z)."""
    lo, hi = 0.0, 2.0
    for _ in range(200):
        t = (lo + hi) / 2
        yt = 50 + vy0 * t + 0.5 * ay * t * t
        lo, hi = (t, hi) if yt > y else (lo, t)
    return x0 + vx0 * t + 0.5 * ax * t * t, z0 + vz0 * t + 0.5 * az * t * t


@settings(max_examples=30)
@given(
    st.floats(-130, -110),
    st.floats(-8, 8),
    st.floats(-8, 2),
    st.floats(-20, 20),
    st.floats(20, 32),
    st.floats(-35, -10),
)
def test_projection_matches_simulation(
    vy0: float, vx0: float, vz0: float, ax: float, ay: float, az: float
) -> None:
    x_f, z_f = _simulate(-1.0, 5.5, vx0, vy0, vz0, ax, ay, az, 17 / 12)
    x_m, z_m = _simulate(-1.0, 5.5, vx0, vy0, vz0, ax, ay, az, (17 - 8.5) / 12)
    px, pz = Z.project_to_plane(
        *(np.array([v]) for v in (x_f, z_f, vx0, vy0, vz0, ax, ay, az)), depth_in=8.5
    )
    assert px[0] == pytest.approx(x_m, abs=1e-6)
    assert pz[0] == pytest.approx(z_m, abs=1e-6)
    zero = Z.project_to_plane(
        *(np.array([v]) for v in (x_f, z_f, vx0, vy0, vz0, ax, ay, az)), depth_in=0.0
    )
    assert zero[1][0] == pytest.approx(z_f)


def test_add_zone_features_planes() -> None:
    df = pl.DataFrame(
        {
            "plate_x": [0.0],
            "plate_z": [3.0],
            "abs_top_ft": [TOP],
            "abs_bot_ft": [BOT],
            "stand": ["R"],
            "vx0": [0.0],
            "vy0": [-130.0],
            "vz0": [-6.0],
            "ax": [0.0],
            "ay": [28.0],
            "az": [-30.0],
        }
    )
    a = Z.add_zone_features(df, ZR, project=False)
    m = Z.add_zone_features(df, ZR, project=np.array([True]))
    assert a["zz_ft"].item() == 3.0
    assert m["zz_ft"].item() < 3.0  # descending pitch is lower at mid-plate


def test_wrong_call_dist() -> None:
    dd = np.array([2.0, -2.0, 2.0, -2.0])
    k = np.array([True, True, False, False])
    assert wrong_call_dist(dd, k).tolist() == [2.0, -2.0, -2.0, 2.0]


def test_called_pitches_drops_unprojectable() -> None:
    """Pre-2026 pitches are projected to mid-plate from their trajectory; one without it
    must be dropped, not passed on with a NaN distance (crashed the full-mode overturn fit)."""
    from datetime import date

    from absengine.overturn.labels import called_pitches
    from tests.conftest import frame, row

    rows = [
        row(season=2025, game_date=date(2025, 5, 1), pitch_number=1),
        row(season=2025, game_date=date(2025, 5, 1), pitch_number=2, vx0=None),
    ]
    df = frame(rows).with_columns(pl.col("vx0").cast(pl.Float64))
    heights = pl.DataFrame({"player_id": [10], "height_in": [72.0], "height_source": ["listed"]})
    umps = pl.DataFrame({"game_pk": [1], "umpire_id": [99]})
    out = called_pitches(df.lazy(), None, heights, umps, abs_rules(), 2026)
    assert out["pitch_number"].to_list() == [1]
    assert out["edge_dist_in"].is_finite().all()
