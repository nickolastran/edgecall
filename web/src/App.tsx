import { useEffect, useMemo, useState } from "react";
import { Engine, nearestEdge, zoneDistance, BALL_R, PLATE_HALF_W, type CI, type EngineData, type Result, type Situation } from "./engine";

const API = import.meta.env.VITE_API_URL as string | undefined;
const ZONE_HALF_H = 9.5; // 19 in tall zone (typical certified height ~71 in)
const VIEW = 18; // svg half-extent, inches

interface Form {
  inning: number; isTop: boolean; outs: number; base: number; balls: number; strikes: number;
  away: number; home: number; call: "strike" | "ball"; x: number; z: number; stand: "R" | "L";
  left: number; umpId: number | null;
}

const START: Form = {
  inning: 7, isTop: false, outs: 1, base: 2, balls: 2, strikes: 2, away: 3, home: 3,
  call: "strike", x: 10.2, z: -3, stand: "R", left: 1, umpId: null,
};

export default function App() {
  const [data, setData] = useState<EngineData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/engine.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e: Error) => setErr(`Couldn't load data/engine.json (${e.message}). Run \`make export\` and reload.`));
  }, []);
  if (err) return <main className="shell"><p className="error">{err}</p></main>;
  if (!data) return <main className="shell"><p className="muted">Loading the model…</p></main>;
  return <Tool data={data} />;
}

function Tool({ data }: { data: EngineData }) {
  const engine = useMemo(() => new Engine(data), [data]);
  const [f, setF] = useState<Form>(START);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((o) => ({ ...o, [k]: v }));

  const d = zoneDistance(f.x, f.z, ZONE_HALF_H);
  const edge = nearestEdge(f.x, f.z, ZONE_HALF_H, f.stand);
  const ump = data.umpires.find((u) => u.id === f.umpId) ?? null;
  const sit: Situation = {
    inning: f.inning, is_top: f.isTop, outs: f.outs, base: f.base, balls: f.balls, strikes: f.strikes,
    score_diff_home: f.home - f.away, call: f.call, d_perc: d, edge, challenges_left: f.left,
    ump_mean: ump?.edges[edge]?.mean ?? 0,
  };
  const local = engine.evaluate(sit);
  const remote = useRemote(sit, f, edge);
  const r = remote ?? local;

  return (
    <main className="shell">
      <header className="masthead">
        <h1>EdgeCall</h1>
        <p>Should you challenge that call? Put the pitch where you saw it, set the game situation, and get the call based on the 2026 ABS season.</p>
      </header>

      <section className="desk">
        <div className="zone-panel">
          <Zone f={f} onPick={(x, z) => setF((o) => ({ ...o, x, z }))} />
          <div className="seg" role="group" aria-label="Umpire's call">
            <span>Umpire called it a</span>
            <Toggle value={f.call} options={[["strike", "Strike"], ["ball", "Ball"]]} onChange={(v) => set("call", v)} />
          </div>
          <p className="readout">
            Your read: <b>{Math.abs(d).toFixed(1)} in {d > 0 ? "outside" : "inside"}</b> the zone, at the {edge} edge.
            {" "}{r.side === "offense" ? "The batter decides." : "The catcher or pitcher decides."}
          </p>
        </div>

        <div className="controls">
          <Field label="Inning">
            <Stepper value={f.inning} min={1} max={15} onChange={(v) => set("inning", v)} />
            <Toggle value={f.isTop ? "top" : "bot"} options={[["top", "Top"], ["bot", "Bottom"]]} onChange={(v) => set("isTop", v === "top")} />
          </Field>
          <Field label="Score">
            <label className="score">Away <input type="number" min={0} max={30} value={f.away} onChange={(e) => set("away", +e.target.value || 0)} /></label>
            <label className="score">Home <input type="number" min={0} max={30} value={f.home} onChange={(e) => set("home", +e.target.value || 0)} /></label>
          </Field>
          <Field label="Count">
            <Toggle value={String(f.balls)} options={[0, 1, 2, 3].map((b) => [String(b), `${b}`])} onChange={(v) => set("balls", +v)} ariaLabel="Balls" />
            <span className="dash">–</span>
            <Toggle value={String(f.strikes)} options={[0, 1, 2].map((s) => [String(s), `${s}`])} onChange={(v) => set("strikes", +v)} ariaLabel="Strikes" />
          </Field>
          <Field label="Outs">
            <Toggle value={String(f.outs)} options={[0, 1, 2].map((o) => [String(o), `${o}`])} onChange={(v) => set("outs", +v)} ariaLabel="Outs" />
          </Field>
          <Field label="Runners">
            <Bases base={f.base} onChange={(b) => set("base", b)} />
          </Field>
          <Field label={`Challenges left (${r.side === "offense" ? "batting" : "fielding"} team)`}>
            <Toggle value={String(f.left)} options={[0, 1, 2, 3].map((c) => [String(c), `${c}`])} onChange={(v) => set("left", +v)} ariaLabel="Challenges left" />
          </Field>
          <Field label="Batter">
            <Toggle value={f.stand} options={[["R", "Righty"], ["L", "Lefty"]]} onChange={(v) => set("stand", v)} />
          </Field>
          <Field label="Plate umpire">
            <select value={f.umpId ?? ""} onChange={(e) => set("umpId", e.target.value ? +e.target.value : null)}>
              <option value="">Average umpire</option>
              {data.umpires.map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
            </select>
          </Field>
        </div>

        <Verdict r={r} remote={remote !== null} />
      </section>

      <ThresholdMap engine={engine} sit={sit} p={r.p_overturn} />
      {data.backtest && <Backtest bt={data.backtest} />}
      <Umpires data={data} selected={f.umpId} onPick={(id) => set("umpId", id)} />

      <footer className="foot">
        {data.meta.sample ? "Fit on 2026 data only (sample mode). " : ""}
        Model data generated {data.meta.generated.slice(0, 10)}.
        {data.windows?.test && <> Backtest window {data.windows.test.from} to {data.windows.test.to}.</>}
      </footer>
    </main>
  );
}

function useRemote(sit: Situation, f: Form, edge: string): Result | null {
  const [res, setRes] = useState<Result | null>(null);
  const body = JSON.stringify({ ...sit, outs: sit.outs, edge, stand: f.stand, umpire_id: f.umpId, ump_mean: undefined });
  useEffect(() => {
    if (!API) return;
    const ctl = new AbortController();
    fetch(`${API.replace(/\/$/, "")}/evaluate`, { method: "POST", headers: { "Content-Type": "application/json" }, body, signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then(setRes)
      .catch(() => setRes(null));
    return () => ctl.abort();
  }, [body]);
  return API ? res : null;
}

function Zone({ f, onPick }: { f: Form; onPick: (x: number, z: number) => void }) {
  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.type === "pointermove" && e.buttons !== 1) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 2 * VIEW - VIEW;
    const z = VIEW - ((e.clientY - r.top) / r.height) * 2 * VIEW;
    onPick(Math.round(x * 10) / 10, Math.round(z * 10) / 10);
  };
  const key = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 1 : 0.25;
    const m: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
    const v = m[e.key];
    if (v) { e.preventDefault(); onPick(f.x + v[0], f.z + v[1]); }
  };
  const H = ZONE_HALF_H, W = PLATE_HALF_W;
  return (
    <svg className="zone" viewBox={`${-VIEW} ${-VIEW} ${2 * VIEW} ${2 * VIEW}`} onPointerDown={pick} onPointerMove={pick}
      tabIndex={0} onKeyDown={key} role="slider" aria-label="Pitch location (catcher's view). Arrow keys move it."
      aria-valuetext={`${f.x.toFixed(1)} in horizontal, ${f.z.toFixed(1)} in vertical`}>
      <g transform="scale(1,-1)">
        {[-W / 3, W / 3].map((x) => <line key={x} className="grid" x1={x} x2={x} y1={-H} y2={H} />)}
        {[-H / 3, H / 3].map((y) => <line key={y} className="grid" x1={-W} x2={W} y1={y} y2={y} />)}
        <rect className="abs" x={-W} y={-H} width={2 * W} height={2 * H} />
        <rect className="reach" x={-W - BALL_R} y={-H - BALL_R} width={2 * (W + BALL_R)} height={2 * (H + BALL_R)} rx={BALL_R} />
        <path className="plate" d={`M${-W} ${-VIEW + 1.5} h${2 * W} v-1.5 l${-W} -2.5 l${-W} 2.5 z`} />
        <circle className={`pitch ${f.call}`} cx={f.x} cy={f.z} r={BALL_R} />
      </g>
      <text className="tag" x={-VIEW + 1} y={-VIEW + 2.5}>{f.stand === "R" ? "Inside ←" : "← Outside"}</text>
      <text className="tag" x={VIEW - 1} y={-VIEW + 2.5} textAnchor="end">{f.stand === "R" ? "→ Outside" : "Inside →"}</text>
    </svg>
  );
}

function Verdict({ r, remote }: { r: Result; remote: boolean }) {
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const none = r.option_value === null;
  return (
    <aside className={`verdict ${r.challenge ? "go" : "hold"}`} aria-live="polite">
      <p className="call">{none ? "No challenges left" : r.challenge ? "Challenge" : "Hold"}</p>
      <div className="meter" aria-hidden>
        <div className="fill" style={{ width: pct(r.p_overturn) }} />
        {!none && <div className="mark" style={{ left: pct(Math.min(r.p_threshold, 1)) }} />}
      </div>
      <p className="why">
        {pct(r.p_overturn)} chance it gets overturned
        {!none && <>, and you need {pct(r.p_threshold)} to break even.</>}
      </p>
      <dl>
        <dt>Win if overturned</dt><dd>+{(r.gain_wins * 100).toFixed(1)}% win prob, {r.gain_runs.toFixed(2)} runs</dd>
        {!none && <><dt>Cost of losing the challenge</dt><dd>{(r.option_value! * 100).toFixed(1)}% win prob</dd></>}
        {r.ev_wins !== null && <><dt>Expected value of challenging</dt><dd>{r.ev_wins >= 0 ? "+" : ""}{(r.ev_wins * 100).toFixed(2)}% win prob</dd></>}
      </dl>
      {remote && <p className="src">Scored by the live API with the full model.</p>}
    </aside>
  );
}

function ThresholdMap({ engine, sit, p }: { engine: Engine; sit: Situation; p: number }) {
  const diffs = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
  const halves = Array.from({ length: 18 }, (_, i) => ({ inning: Math.floor(i / 2) + 1, top: i % 2 === 0 }));
  const homeSide = (top: boolean) => (sit.call === "strike" ? !top : top); // challenging team is home?
  const myDiff = (homeSide(sit.is_top) ? 1 : -1) * sit.score_diff_home;
  return (
    <section className="block">
      <h2>When would this pitch be worth it?</h2>
      <p className="lede">
        Each cell is the break-even overturn chance for the same pitch, count, runners and challenges left,
        at a different point in the game. Highlighted cells are where your {Math.round(p * 100)}% read clears the bar.
      </p>
      <div className="heat-wrap">
        <table className="heat">
          <thead>
            <tr><th scope="col">Half</th>{diffs.map((x) => <th key={x} scope="col">{x > 0 ? `Up ${x}` : x < 0 ? `Down ${-x}` : "Tied"}</th>)}</tr>
          </thead>
          <tbody>
            {halves.map(({ inning, top }) => (
              <tr key={`${inning}${top}`}>
                <th scope="row">{top ? "Top" : "Bot"} {inning}</th>
                {diffs.map((dt) => {
                  const homeDiff = homeSide(top) ? dt : -dt;
                  const impossible = !top && inning >= 9 && homeDiff > 0;
                  if (impossible) return <td key={dt} className="na">·</td>;
                  const r = engine.evaluate({ ...sit, inning, is_top: top, score_diff_home: homeDiff });
                  const here = inning === sit.inning && top === sit.is_top && dt === Math.max(-4, Math.min(4, myDiff));
                  return (
                    <td key={dt} className={`${p > r.p_threshold && r.option_value !== null ? "clear" : ""} ${here ? "here" : ""}`}
                      style={{ ["--t" as string]: r.p_threshold }}>
                      {r.option_value === null ? "–" : Math.round(r.p_threshold * 100)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">Score is from the challenging team's side. Outlined cell = the situation you set.</p>
    </section>
  );
}

const POLICY_NAMES: Record<string, string> = {
  never: "Never challenge",
  dp_perceived: "EdgeCall policy",
  human: "What 2026 players actually did",
  dp_oracle: "Perfect information (ceiling)",
};
const policyName = (k: string) => POLICY_NAMES[k] ?? k.replace(/^naive_(.+)in$/, "Challenge anything that looks $1+ in off");

function Backtest({ bt }: { bt: NonNullable<EngineData["backtest"]> }) {
  const rows = Object.entries(bt.policies).sort((a, b) => (b[1].wins_all as CI).mean - (a[1].wins_all as CI).mean);
  const max = Math.max(...rows.map(([, v]) => (v.wins_all as CI).hi));
  const hvd = bt.human_vs_dp;
  return (
    <section className="block">
      <h2>How much is a good policy worth?</h2>
      <p className="lede">
        Every adverse call from the held-out half of 2026 ({bt.n_team_games.toLocaleString()} team-games), replayed under each policy with the real challenge rules.
        Wins and runs are per team over a 162-game season, with 95% intervals from resampling games.
      </p>
      <div className="bt">
        {rows.map(([k, v]) => {
          const w = v.wins_all as CI, runs = v.runs_all as CI;
          return (
            <div key={k} className={`bt-row ${k === "dp_perceived" ? "ours" : ""} ${k === "dp_oracle" ? "ceiling" : ""}`}>
              <span className="bt-name">{policyName(k)}</span>
              <span className="bt-bar" aria-hidden>
                <span className="ci" style={{ left: `${(w.lo / max) * 100}%`, width: `${((w.hi - w.lo) / max) * 100}%` }} />
                <span className="pt" style={{ left: `${(w.mean / max) * 100}%` }} />
              </span>
              <span className="bt-num">{w.mean.toFixed(1)} wins <small>({w.lo.toFixed(1)}–{w.hi.toFixed(1)})</small></span>
              <span className="bt-sub">{runs.mean.toFixed(0)} runs, {v.challenges_per_team_game.toFixed(1)} challenges a game, {(v.overturn_rate * 100).toFixed(0)}% overturned</span>
            </div>
          );
        })}
      </div>
      {hvd.human_only && hvd.dp_only && (
        <p className="note">
          Where players and the policy disagreed: players made {hvd.human_only.n.toLocaleString()} challenges the policy would have skipped
          (ABS would overturn {pct0(hvd.human_only.p_overturn_abs)}), and passed on {hvd.dp_only.n.toLocaleString()} the policy would have made
          ({pct0(hvd.dp_only.p_overturn_abs)} would have been overturned, worth more per flip on average).
        </p>
      )}
      <p className="note">
        "Perfect information" knows the tracked location, so it's an upper bound no player can reach. The EdgeCall policy only uses
        a read as noisy as players' real reads, fit from who actually challenged in 2026.
      </p>
    </section>
  );
}
const pct0 = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);

function Umpires({ data, selected, onPick }: { data: EngineData; selected: number | null; onPick: (id: number) => void }) {
  const [q, setQ] = useState("");
  const list = data.umpires.filter((u) => u.name.toLowerCase().includes(q.toLowerCase()));
  const edges = ["top", "bottom", "inside", "outside"] as const;
  const fmt = (v?: number) => (v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`);
  return (
    <section className="block">
      <h2>Plate umpires</h2>
      <p className="lede">
        How much each umpire's zone runs bigger (+) or smaller (−) than average on each edge, as a shift in the log-odds of a strike call
        on borderline pitches. A tight umpire makes strike calls on that edge more likely to be wrong, so batters' challenges are more likely to win.
        Pick one to use them above.
      </p>
      <input className="search" type="search" placeholder="Find an umpire" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Find an umpire" />
      <div className="ump-wrap">
        <table className="umps">
          <thead><tr><th scope="col">Umpire</th>{edges.map((e) => <th key={e} scope="col">{e[0].toUpperCase() + e.slice(1)}</th>)}<th scope="col">Pitches</th></tr></thead>
          <tbody>
            {list.map((u) => (
              <tr key={u.id} className={u.id === selected ? "sel" : ""}>
                <th scope="row"><button type="button" onClick={() => onPick(u.id)}>{u.name}</button></th>
                {edges.map((e) => <td key={e} className={(u.edges[e]?.mean ?? 0) > 0.15 ? "big" : (u.edges[e]?.mean ?? 0) < -0.15 ? "small" : ""}>{fmt(u.edges[e]?.mean)}</td>)}
                <td>{u.n.toLocaleString()}</td>
              </tr>
            ))}
            {list.length === 0 && <tr><td colSpan={6} className="muted">No umpire matches “{q}”.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="field"><span className="label">{label}</span><div className="row">{children}</div></div>;
}

function Toggle<T extends string>({ value, options, onChange, ariaLabel }: { value: T; options: [T, string][]; onChange: (v: T) => void; ariaLabel?: string }) {
  return (
    <div className="toggle" role="radiogroup" aria-label={ariaLabel}>
      {options.map(([v, l]) => (
        <button key={v} type="button" role="radio" aria-checked={v === value} className={v === value ? "on" : ""} onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  );
}

function Stepper({ value, min, max, onChange }: { value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <div className="stepper">
      <button type="button" aria-label="Previous inning" onClick={() => onChange(Math.max(min, value - 1))}>−</button>
      <span>{value}</span>
      <button type="button" aria-label="Next inning" onClick={() => onChange(Math.min(max, value + 1))}>+</button>
    </div>
  );
}

function Bases({ base, onChange }: { base: number; onChange: (b: number) => void }) {
  const bag = (bit: number, label: string, x: number, y: number) => (
    <button type="button" key={bit} className={`bag ${base & bit ? "on" : ""}`} style={{ left: x, top: y }}
      aria-pressed={!!(base & bit)} aria-label={label} onClick={() => onChange(base ^ bit)} />
  );
  return <div className="diamond">{bag(2, "Runner on second", 22, 0)}{bag(4, "Runner on third", 0, 22)}{bag(1, "Runner on first", 44, 22)}</div>;
}
