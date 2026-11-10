// Port of absengine's decision math (gamestate.after_call, re24.value_after,
// win_expectancy.we_after_call, ev.flip_gain/threshold, dp.DPResult.option_value).
// Parity with Python is checked against `golden` cases in engine.test.ts.

export type Edge = "top" | "bottom" | "inside" | "outside";

export interface Situation {
  inning: number;
  is_top: boolean;
  outs: number;
  base: number; // bit-packed: 1B=1, 2B=2, 3B=4
  balls: number;
  strikes: number;
  score_diff_home: number;
  call: "strike" | "ball";
  d_perc: number; // perceived signed distance to the ABS zone, inches; > 0 = outside
  edge: Edge;
  challenges_left: number;
  ump_mean?: number; // umpire tendency on this edge (logit shift), 0 = average
}

export interface Result {
  side: "offense" | "defense";
  p_overturn: number;
  gain_wins: number;
  gain_runs: number;
  option_value: number | null;
  p_threshold: number;
  ev_wins: number | null;
  challenge: boolean;
}

export interface CI { mean: number; lo: number; hi: number }
export interface PolicyRow {
  challenges_per_team_game: number;
  overturn_rate: number;
  [k: string]: number | CI;
}
export interface Umpire {
  id: number;
  name: string;
  n: number;
  edges: Partial<Record<Edge, { mean: number; sd: number }>>;
}

export interface EngineData {
  meta: { sample: boolean; generated: string };
  rules: { challenges_per_team: number; retain_on_success: boolean; regulation_innings: number };
  count_re: number[][];
  we: { p: number[]; shape: number[]; max_inning: number; max_diff: number; regulation: number; ghost_runner: boolean };
  dp: { V: Record<string, number[][][]>; cont: Record<string, number[][][]>; edges: number[]; max_c: number };
  grid: { edges: Edge[]; d: number[]; ump: number[]; p: number[] };
  umpires: Umpire[];
  backtest: {
    n_team_games: number;
    policies: Record<string, PolicyRow>;
    human_vs_dp: Record<string, { n: number; p_overturn_abs: number | null; mean_gain_wins: number | null }>;
  } | null;
  overturn_eval: { lgbm_isotonic: { n: number; mean_p: number; mean_y: number; brier: number } } | null;
  windows: Record<string, { n: number; from: string; to: string }> | null;
  golden: { in: Situation; out: Result }[];
}

const N_OUTS = 3;

interface Transition { paOver: boolean; balls: number; strikes: number; base: number; outs: number; runs: number }

export function afterCall(balls: number, strikes: number, base: number, outs: number, isStrike: boolean): Transition {
  if (isStrike) {
    if (strikes === 2) return { paOver: true, balls: 0, strikes: 0, base, outs: outs + 1, runs: 0 };
    return { paOver: false, balls, strikes: strikes + 1, base, outs, runs: 0 };
  }
  if (balls === 3) {
    const on1 = !!(base & 1), on2 = !!(base & 2), on3 = !!(base & 4);
    const nb = 1 | ((on2 || on1) ? 2 : 0) | ((on3 || (on1 && on2)) ? 4 : 0);
    return { paOver: true, balls: 0, strikes: 0, base: nb, outs, runs: on1 && on2 && on3 ? 1 : 0 };
  }
  return { paOver: false, balls: balls + 1, strikes, base, outs, runs: 0 };
}

const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export class Engine {
  constructor(readonly d: EngineData) {}

  valueAfter(balls: number, strikes: number, base: number, outs: number, strike: boolean): number {
    const t = afterCall(balls, strikes, base, outs, strike);
    if (t.outs >= N_OUTS) return t.runs;
    return t.runs + this.d.count_re[t.balls * 3 + t.strikes][t.base * N_OUTS + t.outs];
  }

  weLookup(inning: number, isTop: boolean, outs: number, base: number, diff: number): number {
    const w = this.d.we, [, s1, s2, s3, s4] = w.shape;
    const i = Math.min(inning, w.max_inning) - 1;
    const d = clip(diff, -w.max_diff, w.max_diff) + w.max_diff;
    return w.p[(((i * s1 + (isTop ? 1 : 0)) * s2 + outs) * s3 + base) * s4 + d];
  }

  private startOfHalf(inning: number, isTop: boolean, diff: number): number {
    const w = this.d.we;
    const base = w.ghost_runner && inning > w.regulation ? 2 : 0;
    return this.weLookup(inning, isTop, 0, base, diff);
  }

  weState(inning: number, isTop: boolean, outs: number, base: number, diff: number, delta: number): number {
    const reg = this.d.we.regulation;
    if (!isTop && inning >= reg && diff > 0) return 1;
    if (outs >= N_OUTS) {
      if (isTop) {
        if (inning >= reg && diff > 0) return 1;
        return this.startOfHalf(inning, false, diff);
      }
      if (inning >= reg && diff !== 0) return diff > 0 ? 1 : 0;
      return this.startOfHalf(inning + 1, true, diff);
    }
    let p = this.weLookup(inning, isTop, outs, base, diff);
    if (delta) {
      const slope = (this.weLookup(inning, isTop, outs, base, diff + 1) - this.weLookup(inning, isTop, outs, base, diff - 1)) / 2;
      p += slope * delta * (isTop ? -1 : 1);
    }
    return clip(p, 0, 1);
  }

  weAfterCall(x: Situation, strike: boolean): number {
    const t = afterCall(x.balls, x.strikes, x.base, x.outs, strike);
    const diff = x.score_diff_home + (x.is_top ? -t.runs : t.runs);
    let delta = 0;
    if (!t.paOver && t.outs < N_OUTS) {
      const bo = t.base * N_OUTS + t.outs;
      delta = this.d.count_re[t.balls * 3 + t.strikes][bo] - this.d.count_re[0][bo];
    }
    return this.weState(x.inning, x.is_top, t.outs, t.base, diff, delta);
  }

  flipGain(x: Situation): [number, number] {
    const runs = this.valueAfter(x.balls, x.strikes, x.base, x.outs, false) - this.valueAfter(x.balls, x.strikes, x.base, x.outs, true);
    const offWins = (this.weAfterCall(x, false) - this.weAfterCall(x, true)) * (x.is_top ? -1 : 1);
    return [Math.abs(runs), Math.abs(offWins)];
  }

  optionValue(side: "home" | "away", inning: number, isTop: boolean, outs: number, diffTeam: number, c: number): number {
    if (c <= 0) return Infinity;
    const V = this.d.dp.V[side], C = this.d.dp.cont[side];
    const t = Math.min(2 * (inning - 1) + (isTop ? 0 : 1), V.length - 2);
    const b = this.d.dp.edges.filter((e) => e < diffTeam).length; // searchsorted(side="left")
    c = Math.min(c, this.d.dp.max_c);
    const frac = (3 - outs) / 3;
    return frac * (V[t][b][c] - V[t][b][c - 1]) + (1 - frac) * (C[t][b][c] - C[t][b][c - 1]);
  }

  pOverturn(call: "strike" | "ball", edge: Edge, balls: number, strikes: number, d: number, ump = 0): number {
    const g = this.d.grid, nd = g.d.length, nu = g.ump.length;
    const base = ((((call === "strike" ? 1 : 0) * 4 + g.edges.indexOf(edge)) * 4 + balls) * 3 + strikes) * nd * nu;
    const at = (i: number, j: number) => g.p[base + i * nu + j];
    const [i0, i1, fd] = bracket(g.d, d), [j0, j1, fu] = bracket(g.ump, ump);
    const lo = at(i0, j0) * (1 - fu) + at(i0, j1) * fu;
    const hi = at(i1, j0) * (1 - fu) + at(i1, j1) * fu;
    return lo * (1 - fd) + hi * fd;
  }

  evaluate(x: Situation): Result {
    const offense = x.call === "strike";
    const teamIsHome = offense ? !x.is_top : x.is_top;
    const diffTeam = teamIsHome ? x.score_diff_home : -x.score_diff_home;
    const [gRuns, gWins] = this.flipGain(x);
    const p = this.pOverturn(x.call, x.edge, x.balls, x.strikes, x.d_perc, x.ump_mean ?? 0);
    const has = x.challenges_left > 0;
    const ov = this.optionValue(teamIsHome ? "home" : "away", x.inning, x.is_top, x.outs, diffTeam, x.challenges_left);
    const thr = has ? (gWins + ov > 0 ? ov / (gWins + ov) : 1) : 1;
    return {
      side: offense ? "offense" : "defense",
      p_overturn: p,
      gain_wins: gWins,
      gain_runs: gRuns,
      option_value: has ? ov : null,
      p_threshold: thr,
      ev_wins: has ? p * gWins - (1 - p) * ov : null,
      challenge: has && p > thr,
    };
  }
}

function bracket(xs: number[], x: number): [number, number, number] {
  const n = xs.length;
  if (x <= xs[0]) return [0, 0, 0];
  if (x >= xs[n - 1]) return [n - 1, n - 1, 0];
  let i = 0;
  while (xs[i + 1] < x) i++;
  return [i, i + 1, (x - xs[i]) / (xs[i + 1] - xs[i])];
}

// ABS zone geometry (features/zone.py), catcher's view, inches from zone center.
export const PLATE_HALF_W = 8.5;
export const BALL_R = 1.45;

export function zoneDistance(x: number, z: number, halfH: number): number {
  const qx = Math.abs(x) - PLATE_HALF_W, qz = Math.abs(z) - halfH;
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qz, 0));
  const inside = Math.min(Math.max(qx, qz), 0);
  return outside + inside - BALL_R;
}

export function nearestEdge(x: number, z: number, halfH: number, stand: "R" | "L"): Edge {
  const qx = Math.abs(x) - PLATE_HALF_W, qz = Math.abs(z) - halfH;
  if (qz > qx) return z >= 0 ? "top" : "bottom";
  return (stand === "R" ? x < 0 : x > 0) ? "inside" : "outside";
}
