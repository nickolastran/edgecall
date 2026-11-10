"""Backward-induction value of holding challenges, and the resulting decision thresholds.

Per team side (home/away), state = (half index t, own score-diff bucket b, challenges c).
Within a half, each called pitch is a step; a step is challengeable for the team iff the
call went against it (strike while batting, ball while fielding). (P(overturn), gain) at a
step is drawn from the empirical distribution at (t, b). Score buckets move between halves
by an empirical Markov kernel with an absorbing "game over" state.

    W_step(c) = E[ max( W(c),  p*(g + W(c or c-1 if no retention)) + (1-p)*W(c-1) ) ]

Option value of the c-th challenge: Δ(c) = V(c) - V(c-1). Challenge iff p*g > (1-p)*Δ
(with retention), i.e. p > Δ / (g + Δ). See DECISIONS.md P1-P4 for approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from absengine.config import AbsRules


def bucket(diff: np.ndarray | int, edges: list[int]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(diff), side="left")


def half_index(inning: np.ndarray | int, is_top: np.ndarray | bool) -> np.ndarray:
    return np.asarray(2 * (np.asarray(inning) - 1) + (~np.asarray(is_top, dtype=bool)).astype(int))


@dataclass
class DPResult:
    V: dict[str, np.ndarray]  # side -> [T+1, B, C+1]  value at START of half t
    cont: dict[str, np.ndarray]  # side -> [T, B, C+1]  value after half t ends
    edges: list[int]
    max_c: int
    regulation: int
    meta: dict[str, object] = field(default_factory=dict)

    def option_value(
        self, side: str, inning: int, is_top: bool, outs: int, diff_team: int, c: int
    ) -> float:
        """Δ(c) mid-half, interpolating between start-of-half and next-half values by the
        fraction of the half left ((3 - outs) / 3)."""
        if c <= 0:
            return float("inf")
        t = int(min(half_index(inning, is_top), self.V[side].shape[0] - 2))
        b = int(bucket(diff_team, self.edges))
        c = min(c, self.max_c)
        frac = (3 - outs) / 3
        v_start, v_end = self.V[side][t, b], self.cont[side][t, b]
        return float(frac * (v_start[c] - v_start[c - 1]) + (1 - frac) * (v_end[c] - v_end[c - 1]))


def _step(W: np.ndarray, p: np.ndarray, g: np.ndarray, retain: bool) -> np.ndarray:
    out = W.copy()
    for c in range(1, len(W)):
        keep = W[c] if retain else W[c - 1]
        chal = p * (g + keep) + (1 - p) * W[c - 1]
        out[c] = np.mean(np.maximum(W[c], chal))
    return out


def solve(
    opps: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
    steps: np.ndarray,
    trans: dict[str, np.ndarray],
    T: int,
    n_buckets: int,
    rules: AbsRules,
    max_c: int,
    edges: list[int],
) -> DPResult:
    """opps[(side, t, b)] = (p, g) samples (zeros where not challengeable);
    steps[t] = called pitches per half; trans[side][t] = [B, B+1] kernel (last col = end)."""
    V: dict[str, np.ndarray] = {}
    CONT: dict[str, np.ndarray] = {}
    reg = rules.regulation_innings
    for side in ("home", "away"):
        v = np.zeros((T + 1, n_buckets, max_c + 1))
        cont = np.zeros((T, n_buckets, max_c + 1))
        for t in range(T - 1, -1, -1):
            nxt = v[t + 1].copy()
            next_inning = (t + 1) // 2 + 1
            if (t + 1) % 2 == 0 and next_inning > reg and rules.extra_inning_grant:
                # entering an extra inning: a team with 0 is granted challenges
                if rules.extra_inning_grant_only_if_empty:
                    nxt[:, 0] = nxt[:, min(rules.extra_inning_grant, max_c)]
                else:
                    g_ = rules.extra_inning_grant
                    nxt = np.concatenate([nxt[:, g_:], np.repeat(nxt[:, -1:], g_, axis=1)], axis=1)
            K = trans[side][t]  # [B, B+1]
            cont[t] = K[:, :n_buckets] @ nxt  # end state has value 0
            for b in range(n_buckets):
                p, g = opps[(side, t, b)]
                W = cont[t, b].copy()
                for _ in range(round(steps[t])):
                    W = _step(W, p, g, rules.retain_on_success)
                v[t, b] = W
        V[side], CONT[side] = v, cont
    return DPResult(V, CONT, edges, max_c, reg)


# --------------------------------------------------------------------------- inputs


def build_inputs(
    called: pl.DataFrame,
    edges: list[int],
    T: int,
    min_samples: int,
    max_samples: int,
    seed: int,
) -> tuple[
    dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, np.ndarray]
]:
    """From historical called pitches with columns: game_pk, inning, is_top, at_bat_number,
    score_diff_home, is_strike_call, p_overturn, gain_wins."""
    rng = np.random.default_rng(seed)
    nb = len(edges) + 1
    df = called.with_columns(
        t=pl.Series(
            np.minimum(half_index(called["inning"].to_numpy(), called["is_top"].to_numpy()), T - 1)
        ),
    )
    # steps per half: called pitches per (game, half), averaged per t
    per_half = df.group_by("game_pk", "t").len()
    steps = np.zeros(T)
    mean_by_t = dict(per_half.group_by("t").agg(pl.col("len").mean()).iter_rows())
    for t in range(T):
        steps[t] = mean_by_t.get(t, np.mean(list(mean_by_t.values())))

    opps: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    trans: dict[str, np.ndarray] = {}
    for side, sign in (("home", 1), ("away", -1)):
        diff_team = df["score_diff_home"].to_numpy() * sign
        b = bucket(diff_team, edges)
        batting = df["is_top"].to_numpy() == (side == "away")
        adverse = np.where(
            batting, df["is_strike_call"].to_numpy(), ~df["is_strike_call"].to_numpy()
        )
        p = np.where(adverse, df["p_overturn"].to_numpy(), 0.0)
        g = np.where(adverse, df["gain_wins"].to_numpy(), 0.0)
        t_arr = df["t"].to_numpy()
        for t in range(T):
            at_t = t_arr == t
            for bb in range(nb):
                m = at_t & (b == bb)
                if m.sum() < min_samples:  # pool buckets when thin (extras, blowouts)
                    m = at_t if at_t.sum() >= min_samples else (t_arr >= min(t, T - 2))
                idx = np.flatnonzero(m)
                if len(idx) > max_samples:
                    idx = rng.choice(idx, max_samples, replace=False)
                opps[(side, t, bb)] = (p[idx], g[idx])
        # bucket transitions between consecutive halves of the same game
        first = (
            df.with_columns(bk=pl.Series(b))
            .sort("game_pk", "t", "at_bat_number")
            .group_by("game_pk", "t", maintain_order=True)
            .agg(pl.col("bk").first())
            .sort("game_pk", "t")
        )
        gp, tt, bk = (first[c].to_numpy() for c in ("game_pk", "t", "bk"))
        counts = np.ones((T, nb, nb + 1)) * 0.01  # tiny smoothing
        for i in range(len(gp)):
            if tt[i] >= T:
                continue
            nxt_same = i + 1 < len(gp) and gp[i + 1] == gp[i] and tt[i + 1] == tt[i] + 1
            counts[tt[i], bk[i], bk[i + 1] if nxt_same else nb] += 1
        trans[side] = counts / counts.sum(-1, keepdims=True)
    return opps, steps, trans
