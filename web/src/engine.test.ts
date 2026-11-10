import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Engine, type EngineData, zoneDistance } from "./engine";

const data = JSON.parse(readFileSync(new URL("../public/data/engine.json", import.meta.url), "utf8")) as EngineData;
const e = new Engine(data);

describe("parity with the Python engine", () => {
  it.each(data.golden.map((g, i) => [i, g] as const))("case %i", (_, g) => {
    const r = e.evaluate(g.in);
    // tables are rounded to 4-5 dp in the export
    expect(r.side).toBe(g.out.side);
    expect(r.gain_runs).toBeCloseTo(g.out.gain_runs, 3);
    expect(r.gain_wins).toBeCloseTo(g.out.gain_wins, 3);
    expect(r.p_overturn).toBeCloseTo(g.out.p_overturn, 3);
    if (g.out.option_value !== null) expect(r.option_value!).toBeCloseTo(g.out.option_value, 3);
    if (Math.abs(g.out.p_overturn - g.out.p_threshold) > 0.01) expect(r.challenge).toBe(g.out.challenge);
  });
});

it("zone distance: center is deep inside, ball grazing the corner is a strike", () => {
  expect(zoneDistance(0, 0, 9.5)).toBeCloseTo(-8.5 - 1.45);
  expect(zoneDistance(8.5 + 1.0, 0, 9.5)).toBeLessThan(0);
  expect(zoneDistance(8.5 + 2.0, 0, 9.5)).toBeGreaterThan(0);
});
