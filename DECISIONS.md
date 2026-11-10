# Decisions

Each entry gives the choice, why, and the alternative considered.

## Data

**D1. Read the platform's raw Statcast Parquet cache, not its Postgres DB.**
`~/stat-sightline`'s `savant.pitches` drops `fielder_2`, `bat_score`, `fld_score` and
`post_bat_score`, all of which this project needs. The raw cache has all 119 Statcast columns.
*Alternative:* query Postgres and re-fetch the missing columns. That means two sources and a join.

**D2. Missing seasons are fetched with the platform's own `fetch_statcast`, into edgecall's cache.**
It runs in the platform venv as a subprocess with `cache_dir=edgecall/data/raw/statcast`, so the platform repo is never written to.
*Alternative:* re-implement the pybaseball pull, which duplicates code.

**D3. Deduplicate on `(game_pk, at_bat_number, pitch_number)`.**
The platform cache has overlapping 3-day windows (e.g. `08-17_08-19` and `08-19_08-21`). Without dedup, 2026 had 1.07M rows instead of the true 717k.

**D4. Umpires come from one schedule call per season** (`/v1/schedule?hydrate=officials`).
Statcast's `umpire` column is 100% null. *Alternative:* about 15k per-game boxscore calls.

**D5. ABS challenge labels come from the MLB Stats API live feed** (`reviewType == "MJ"`).
- They appear on the pitch event, or on the **play** when the challenged pitch ended the PA (ball four or strike three). Those belong to the play's last pitch. Missing them dropped about 25% of challenges, the highest-leverage ones.
- The feed's `call` and Statcast's `description` are the **post-challenge** call. The umpire's original call = the opposite call when `isOverturned`. This was verified on game 822679: an overturned strike shows as `ball` in both sources.
- Raw feeds are cached (about 140 KB/game gz) so parser fixes never need a re-scrape.

**D6. Certified height for the ABS zone.**
2026 Statcast `sz_top`/`sz_bot` are constant per batter with top/bot ratio exactly 53.5/27, so they already are the ABS zone. Height = `sz_top / 0.535`. Batters with no 2026 appearance use MLB listed height (`height_source = listed`).

## Run value / win expectancy

**R1. Bottom-of-9th-or-later half-innings are excluded from RE.**
Walk-offs truncate runs-to-end. This is the standard RE24 convention.

**R2. Count × base-out RE cells are shrunk toward an additive prior RE(bo) + [RE(count) − RE(all)].**
Pseudo-count = `min_state_n`. They are then projected so RE is nondecreasing in balls and nonincreasing in strikes within each base-out state.
*Why:* with one season, raw cells produced negative "ball vs strike" values, e.g. 2-0 with a runner on 3rd. A ball can never hurt the batting team mid-PA, so this projection removes noise rather than signal.
*Alternative:* shrink toward the base-out marginal. That was tried first and still left 4 negative cells.

**R3. Win expectancy = empirical cells shrunk toward a parametric prior, then isotonic in score diff.**
- The prior is a 2-parameter logistic on a Normal-margin z-score (halves left, current RE).
- The pseudo-count is `prior_strength`.
- WE must be nondecreasing in the home lead, so the weighted isotonic projection only removes noise.
- Count enters at lookup time through the local run→win slope: ΔWE ≈ ∂WE/∂diff × ΔRE(count). The table stays the spec's 5-D state.
- Extras: the ghost runner (runner on 2nd, 0 out) starts each extra half-inning.

**R4. Tables applied to 2026 are fit on 2021–2025 only** (`_fit_seasons`). WE calibration is reported on 2026 held out.

## Serving

**S1. The UI is a static Vite + React + TS app (`web/`) instead of Streamlit.**
The user asked for an app deployable with npm or bun. The pipeline exports JSON (policy, thresholds, umpire effects, the P(overturn) grid) and the decision math is ported to TS. `npm run build` / `bun run build` gives a static site for GitHub Pages, Vercel or Netlify. An optional `VITE_API_URL` uses the live FastAPI `/evaluate`.
*Alternative:* Streamlit as specced. It can't deploy through npm/bun and would mean two UIs to maintain.

**S2. No git operations from tooling.** The user manages commits.
