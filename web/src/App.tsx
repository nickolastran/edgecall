import { useEffect, useMemo, useState } from "react";
import { Engine, nearestEdge, zoneDistance, BALL_R, PLATE_HALF_W, type CI, type EngineData, type Player, type PolicyRow, type Result, type Situation } from "./engine";

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
    <>
    <main className="shell">
      <div className="hero">
      <header className="masthead">
        <h1><img className="logo" src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" />EdgeCall</h1>
        <p>Would you challenge that call? Put the pitch where you saw it, set the game situation, and get the call based on the ABS data from the latest season.</p>
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
          <Field label={`Challenges left for the ${r.side === "offense" ? "batting" : "fielding"} team`} hint={r.side === "offense" ? "Batters challenge strike calls." : "Catchers and pitchers challenge ball calls."}>
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
      </div>

      <ThresholdMap engine={engine} sit={sit} p={r.p_overturn} />
      {data.backtest && <Backtest bt={data.backtest} />}
      {data.players && data.players.length > 0 && <Players players={data.players} />}
      <Umpires data={data} selected={f.umpId} onPick={(id) => set("umpId", id)} />

    </main>
    <footer className="foot">
      <div className="foot-in">
        <p>
          {data.meta.sample ? "Fit on 2026 data only (sample mode). " : ""}
          {data.windows?.train && data.windows.calibrate && data.windows.test && (
            <>Trained on {years(data.windows.train)}, calibrated on {years(data.windows.calibrate)}, tested on {data.windows.test.from} to {data.windows.test.to} ({(Object.values(data.windows).reduce((s, w) => s + w.n, 0) / 1e6).toFixed(1)}M pitches). </>
          )}
          Model data generated {data.meta.generated.slice(0, 10)}.
          {" "}Data from MLB Statcast and the MLB Stats API. Not affiliated with MLB.
          {" "}<a href={`${import.meta.env.BASE_URL}privacy.html`}>Privacy</a> · <a href={`${import.meta.env.BASE_URL}terms.html`}>Terms</a>
        </p>
        <nav className="links" aria-label="Nickolas Tran">
          <span>© {new Date().getFullYear()} Nickolas Tran</span>
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} target="_blank" rel="noreferrer">
              <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden>{ICONS[l.icon]}</svg>
              {l.label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
    </>
  );
}

const years = (w: { from: string; to: string }) => {
  const [a, b] = [w.from.slice(0, 4), w.to.slice(0, 4)];
  return a === b ? a : `${a}–${b}`;
};

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
        <Baseball x={f.x} z={f.z} call={f.call} />
      </g>
      <text className="tag" x={-VIEW + 1} y={-VIEW + 2.5}>{f.stand === "R" ? "Inside ←" : "← Outside"}</text>
      <text className="tag" x={VIEW - 1} y={-VIEW + 2.5} textAnchor="end">{f.stand === "R" ? "→ Outside" : "Inside →"}</text>
    </svg>
  );
}

// Drawn at true size (radius = BALL_R inches). The ring color shows the umpire's call.
function Baseball({ x, z, call }: { x: number; z: number; call: "strike" | "ball" }) {
  const R = BALL_R;
  const seams = [`M${-0.62 * R} ${-0.78 * R} Q${-0.18 * R} 0 ${-0.62 * R} ${0.78 * R}`, `M${0.62 * R} ${-0.78 * R} Q${0.18 * R} 0 ${0.62 * R} ${0.78 * R}`];
  return (
    <g className="baseball" transform={`translate(${x} ${z}) rotate(-20)`}>
      <circle className={`ring ${call}`} r={R + 0.3} />
      <circle className="hide" r={R} />
      {seams.map((d) => <path key={d} className="seam" d={d} />)}
      {seams.map((d) => <path key={`s${d}`} className="stitch" d={d} />)}
    </g>
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
    <section className="block" id="worth">
      <h2>When would this pitch be worth it?</h2>
      <p className="lede">
        Each cell is the balanced overturn chance for the same pitch, count, runners and challenges left,
        at a different point in the game. Highlighted cells are where your {Math.round(p * 100)}% read meets expectation.
      </p>
      <div className="legend" aria-hidden>
        <span><i className="sw clear" /> Worth challenging: your {Math.round(p * 100)}% read clears the bar</span>
        <span><i className="sw ramp" /> Not worth it: darker means a higher bar</span>
      </div>
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

const LINKS = [
  { label: "GitHub", href: "https://github.com/nickolastran/edgecall", icon: "github" },
  { label: "LinkedIn", href: "https://www.linkedin.com/in/nickolas-tran/", icon: "linkedin" },
  { label: "Website", href: "https://nickolastran.vercel.app/", icon: "website" },
] as const;

const ICONS: Record<(typeof LINKS)[number]["icon"], React.ReactNode> = {
  github: <path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.930 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.300 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />,
  linkedin: <path fill="currentColor" d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />,
  website: (
    <g fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20M2 12h20" />
    </g>
  ),
};

interface Row { key: string; name: string; what: string; v: PolicyRow }

function Backtest({ bt }: { bt: NonNullable<EngineData["backtest"]> }) {
  const [showAll, setShowAll] = useState(false);
  const P = bt.policies;
  const wins = (k: string) => (P[k].wins_all as CI).mean;
  const naive = Object.keys(P).filter((k) => k.startsWith("naive_")).sort((a, b) => wins(b) - wins(a));
  const inches = (k: string) => String(Number(k.replace(/^naive_(.+)in$/, "$1")));
  const rule = (k: string): Row => ({
    key: k, name: `Simple rule: ${inches(k)}+ inches`, what: `Challenge anything that looks at least ${inches(k)} inches off.`, v: P[k],
  });
  const rows: Row[] = [
    { key: "dp_perceived", name: "EdgeCall", what: "Challenge only when the odds clear the break-even line for that moment.", v: P.dp_perceived },
    { key: "human", name: "Real 2026 players", what: "What players actually challenged, and how it turned out.", v: P.human },
    ...(showAll ? naive.map(rule) : naive.slice(0, 1).map(rule)),
    { key: "never", name: "Never challenge", what: "Leave every call alone.", v: P.never },
  ].filter((r) => r.v);
  const max = Math.max(...rows.map((r) => (r.v.wins_all as CI).hi)) * 1.04;
  const edge = wins("dp_perceived"), human = wins("human");
  const hvd = bt.human_vs_dp;
  return (
    <section className="block" id="strategies">
      <h2>How many wins are smart challenges worth?</h2>
      <p className="lede">
        We replayed every close call from the second half of 2026 ({bt.n_team_games.toLocaleString()} team-games) under different challenge strategies,
        using the real rules. Each bar is the extra wins a team would get over a full 162-game season.
      </p>
      <p className="bt-head">
        EdgeCall adds <b>{edge.toFixed(1)} wins</b> a season, <b>{(edge - human).toFixed(1)} more</b> than players managed on their own.
      </p>
      <div className="bt">
        {rows.map(({ key, name, what, v }) => {
          const w = v.wins_all as CI;
          return (
            <div key={key} className={`bt-row ${key === "dp_perceived" ? "ours" : ""}`}>
              <div className="bt-label">
                <span className="bt-name">{name}</span>
                <span className="bt-what">{what}</span>
              </div>
              <div className="bt-track" role="img" aria-label={`${w.mean.toFixed(1)} wins, likely ${w.lo.toFixed(1)} to ${w.hi.toFixed(1)}`}>
                <span className="bt-fill" style={{ width: `${(w.mean / max) * 100}%` }} />
                <span className="bt-range" style={{ left: `${(w.lo / max) * 100}%`, width: `${((w.hi - w.lo) / max) * 100}%` }} />
              </div>
              <div className="bt-val">
                <span className="bt-num">{w.mean.toFixed(1)}<small> wins</small></span>
                <span className="bt-sub">
                  {key === "never" ? "no challenges" : `${v.challenges_per_team_game.toFixed(1)} challenges a game, ${Math.round(v.overturn_rate * 100)}% won`}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      {naive.length > 1 && (
        <button type="button" className="bt-more" onClick={() => setShowAll(!showAll)} aria-expanded={showAll}>
          {showAll ? "Show fewer simple rules" : `Show all ${naive.length} simple rules`}
        </button>
      )}
      <div className="bt-notes">
        {hvd.human_only && hvd.dp_only && (
          <p>
            <b>Where players went wrong.</b> They spent {hvd.human_only.n.toLocaleString()} challenges that EdgeCall would have saved, mostly on
            low-stakes calls, and passed on {hvd.dp_only.n.toLocaleString()} that it would have made in bigger moments.
          </p>
        )}
        {P.dp_oracle && (
          <p>
            <b>The ceiling.</b> Knowing exactly where every pitch crossed would be worth {wins("dp_oracle").toFixed(1)} wins.
            No player sees that well, so EdgeCall assumes a read as fuzzy as real players' reads.
          </p>
        )}
        <p className="fine">The thin line on each bar is the likely range (95% interval).</p>
      </div>
    </section>
  );
}

type SortKey = "net_wins" | "challenges" | "success" | "wins_gained" | "wins_spent" | "runs_gained";
const ROLES: [string, string][] = [["all", "Everyone"], ["batter", "Batters"], ["catcher", "Catchers"], ["pitcher", "Pitchers"]];

function Players({ players }: { players: Player[] }) {
  const [role, setRole] = useState("all");
  const [team, setTeam] = useState("all");
  const teams = useMemo(() => [...new Set(players.map((p) => p.team))].sort(), [players]);
  const [minN, setMinN] = useState(10);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("net_wins");
  const [desc, setDesc] = useState(true);
  const val = (p: Player, k: SortKey) => (k === "success" ? p.overturned / p.challenges : p[k]);
  const list = players
    .filter((p) => (role === "all" || p.role === role) && (team === "all" || p.team === team) && p.challenges >= minN && p.name.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => (desc ? val(b, sort) - val(a, sort) : val(a, sort) - val(b, sort)));
  const byRole = (r: string) => {
    const g = players.filter((p) => p.role === r);
    const n = g.reduce((t, p) => t + p.challenges, 0), w = g.reduce((t, p) => t + p.overturned, 0);
    return { n, rate: n ? w / n : 0, net: g.reduce((t, p) => t + p.net_wins, 0) };
  };
  const head = (k: SortKey, label: string) => (
    <th scope="col" aria-sort={sort === k ? (desc ? "descending" : "ascending") : "none"}>
      <button type="button" className={sort === k ? "on" : ""} onClick={() => (sort === k ? setDesc(!desc) : (setSort(k), setDesc(true)))}>
        {label}{sort === k ? (desc ? " ▾" : " ▴") : ""}
      </button>
    </th>
  );
  const w = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
  const c = byRole("catcher"), b = byRole("batter"), pt = byRole("pitcher");
  return (
    <section className="block" id="players">
      <h2>Who gets the most out of their challenges?</h2>
      <p className="lede">
        Every 2026 ABS challenge, from the player who tapped their helmet. Wins gained is the win probability their team picked up from overturned calls;
        wins spent is what each failed challenge cost in future challenges. Catchers won {Math.round(c.rate * 100)}% of {c.n.toLocaleString()} challenges
        {pt.n > 0 ? "," : " and"} batters {Math.round(b.rate * 100)}% of {b.n.toLocaleString()}
        {pt.n > 0 && <>, and pitchers, who rarely challenge, just {Math.round(pt.rate * 100)}% of {pt.n.toLocaleString()} challenges</>}.
      </p>
      <div className="filters">
        <Toggle value={role} options={ROLES} onChange={setRole} ariaLabel="Role" />
        <label className="minn">Team
          <select value={team} onChange={(e) => { setTeam(e.target.value); if (e.target.value !== "all") setMinN(1); }}>
            <option value="all">All teams</option>
            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="minn">At least
          <select value={minN} onChange={(e) => setMinN(+e.target.value)}>
            {[1, 5, 10, 20, 40].map((n) => <option key={n} value={n}>{n}</option>)}
          </select> challenges
        </label>
        <input className="search" type="search" placeholder="Find a player" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Find a player" />
      </div>
      <div className="ump-wrap">
        <table className="umps players">
          <thead>
            <tr>
              <th scope="col">Player</th><th scope="col">Team</th><th scope="col">Role</th>
              {head("challenges", "Challenges")}{head("success", "Won")}{head("wins_gained", "Wins gained")}
              {head("wins_spent", "Wins spent")}{head("net_wins", "Net wins")}{head("runs_gained", "Runs")}
            </tr>
          </thead>
          <tbody>
            {list.map((p) => (
              <tr key={`${p.id}-${p.role}`}>
                <th scope="row">{p.name}</th>
                <td>{p.team}</td>
                <td className="role">{p.role}</td>
                <td>{p.challenges}</td>
                <td>{Math.round((p.overturned / p.challenges) * 100)}%</td>
                <td>{w(p.wins_gained)}</td>
                <td>{p.wins_spent.toFixed(2)}</td>
                <td className={p.net_wins >= 0 ? "big" : "small"}>{w(p.net_wins)}</td>
                <td>{p.runs_gained.toFixed(1)}</td>
              </tr>
            ))}
            {list.length === 0 && <tr><td colSpan={9} className="muted">No players match. Try fewer minimum challenges.</td></tr>}
          </tbody>
        </table>
      </div>
      <p className="note">
        {list.length.toLocaleString()} players shown. Net wins counts what happened, not decision quality: a player who only challenges
        obvious misses in big spots will lead, and one who never challenges shows zero. Pitchers rarely challenge, so their numbers are small.
      </p>
    </section>
  );
}

function Umpires({ data, selected, onPick }: { data: EngineData; selected: number | null; onPick: (id: number) => void }) {
  const [q, setQ] = useState("");
  const list = data.umpires.filter((u) => u.name.toLowerCase().includes(q.toLowerCase()));
  const edges = ["top", "bottom", "inside", "outside"] as const;
  const fmt = (v?: number) => (v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`);
  return (
    <section className="block" id="umpires">
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

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <div className="row">{children}</div>
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
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
  return <div className="diamond">{bag(2, "Runner on second", 38, 6)}{bag(4, "Runner on third", 6, 38)}{bag(1, "Runner on first", 70, 38)}</div>;
}
