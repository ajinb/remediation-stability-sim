"""E0: the analysis checked against simulation (no cluster, no LLM).

E0a  The preprint's measured frontier (remediation-stability-sim, 70 cells) against
     the closed-form boundary, with the loop delay the simulator actually implements.
E0b  Random loop delay: which statistic of the delay distribution predicts stability,
     and does the 3/2-type bound hold, with and without out-of-order landings?
E0c  Constructs under plant-gain mismatch and random delay: every-step proportional,
     Smith predictor (nominal delay model), and event-gated pulse remediation.

Writes JSON results to --out and prints the tables used in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import theory as th  # noqa: E402
from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario  # noqa: E402

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)
GAINS = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9)
DEAD_TIMES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)


# ------------------------------------------------------------------------------- E0a
def e0a(seeds: int = 20) -> dict:
    """Closed-form boundary vs the preprint simulator's frontier.

    The simulator lands an action decided at step t (on the measurement from
    t - sensor_delay) at t + dead_time, where it first moves the plant state
    recorded at that step; the recursion is e[n+1] = e[n] - g e[n - L] with
    L = dead_time + sensor_delay - 1.
    """
    horizon = 400 - SPIKE["disturbance_at"] - 20          # steps available to settle
    cells = []
    for g in GAINS:
        for dt in DEAD_TIMES:
            L = dt + SPIKE["sensor_delay"] - 1
            settled = 0
            for s in range(seeds):
                cfg = ControllerConfig(gain=g, dead_time_mean=dt, act_threshold=5.0)
                r = run_scenario(PlantConfig(**SPIKE), cfg, Constructs(), steps=400, seed=s)
                settled += r.settling_time is not None
            rho = th.spectral_radius(g, L)
            stable = g < th.g_star(L)
            # steps for the envelope to shrink from the disturbance (40) into the
            # settle band (5): a stable cell can still be unobservable in 400 steps
            need = math.log(40 / 5) / math.log(1 / rho) if rho < 1 else math.inf
            cells.append({"gain": g, "dead_time": dt, "L": L, "sim_settled_frac": settled / seeds,
                          "theory_stable": stable, "rho": rho, "steps_to_band": need,
                          "observable": stable and need <= horizon})
    agree_raw = sum((c["sim_settled_frac"] >= 0.5) == c["theory_stable"] for c in cells)
    agree_obs = sum((c["sim_settled_frac"] >= 0.5) == c["observable"] for c in cells)
    return {"cells": cells, "n": len(cells), "agree_theory": agree_raw, "agree_theory_with_horizon": agree_obs}


# ------------------------------------------------------------------------------- E0b
def simulate_random_delay(g: float, sensor: int, lat_steps: np.ndarray, kappa: float = 1.0,
                          guard: bool = False, n: int = 800, band: float = 0.05, window: int = 20):
    """Integrating plant, absolute commands, every-step proportional control.

    Decision k observes e[k - sensor] and lands at k + lat_steps[k]. The command
    in force is the most recently landed one (last writer wins), or, with
    `guard`, the landed command with the newest decision index. Returns
    (settled, settle_step, max_abs_error).
    """
    e = [0.0] * (n + 1)
    e[0] = 1.0
    land = [[] for _ in range(n + int(lat_steps.max()) + 2)]
    for k in range(n):
        land[k + int(lat_steps[k])].append(k)
    in_force = -1
    run = 0
    settle = None
    peak = 1.0
    for t in range(n):
        for k in land[t]:
            if not guard or k > in_force:
                in_force = k
        u = 0.0
        if in_force >= 0:
            src = in_force - sensor
            if src >= 0:
                u = g * e[src]
        nxt = e[t] - kappa * u
        e[t + 1] = nxt
        a = abs(nxt)
        if a > peak:
            peak = a
            if a > 1e6:
                return False, None, math.inf
        if a <= band:
            run += 1
            if run >= window and settle is None:
                settle = t + 2 - window
        else:
            run = 0
            settle = None
    return settle is not None, settle, peak


def e0b(seeds: int = 60) -> dict:
    """Random decision latency: stability vs gain for lognormal latencies."""
    rng = np.random.default_rng(7)
    sensor = 1
    median = 2.0
    out = {"sensor": sensor, "median_latency_steps": median, "seeds": seeds, "horizon": 800, "dists": []}
    gains = np.round(np.arange(0.10, 0.701, 0.025), 3)
    for sigma in (0.25, 0.5, 1.0):
        for cap in (None, 8):
            samples = median * np.exp(rng.normal(0, sigma, size=200000))
            if cap is not None:
                samples = np.minimum(samples, cap)
            stats = {name: int(np.ceil(np.percentile(samples, q))) for name, q in
                     (("median", 50), ("p90", 90), ("p95", 95), ("p99", 99))}
            stats["mean"] = int(np.ceil(samples.mean()))
            stats["max"] = int(np.ceil(samples.max())) if cap is not None else None
            lats = []
            for s in range(seeds):
                lat = np.ceil(median * np.exp(np.random.default_rng(1000 + s).normal(0, sigma, 900))).astype(int)
                lats.append(np.minimum(lat, cap) if cap is not None else lat)
            curve = {}
            for guard in (False, True):
                pts = []
                for g in gains:
                    ok = sum(simulate_random_delay(float(g), sensor, lat, guard=guard)[0] for lat in lats)
                    pts.append((float(g), ok / seeds))
                curve["guard" if guard else "last_writer_wins"] = pts

            def g50(pts):
                for (g0, p0), (g1, p1) in zip(pts, pts[1:]):
                    if p0 >= 0.5 > p1:
                        return g0 + (p0 - 0.5) * (g1 - g0) / (p0 - p1)
                return None

            pred = {f"g_star(L_{k})": th.g_star(sensor + v) for k, v in stats.items() if v is not None}
            if cap is not None:
                pred["varying_delay_bound(K=cap+sensor)"] = th.varying_delay_gain(cap + sensor)
            out["dists"].append({"sigma": sigma, "cap": cap, "latency_stats_steps": stats,
                                 "g50_last_writer_wins": g50(curve["last_writer_wins"]),
                                 "g50_guard": g50(curve["guard"]),
                                 "predictions": pred, "curves": curve})
            print(f"  done sigma={sigma} cap={cap}", flush=True)
    return out


# ------------------------------------------------------------------------------- E0c
def simulate_construct(kind: str, g: float, sensor: int, lat: np.ndarray, kappa: float,
                       L_nominal: int, n: int = 400, band: float = 0.05, window: int = 10):
    """Cycle-accurate abstract loop for one construct; returns (settled, settle_step)."""
    e = np.zeros(n + 1)
    e[0] = 1.0
    if kind == "every_step":
        ok, st, _ = simulate_random_delay(g, sensor, lat, kappa=kappa, guard=True, n=n, band=band, window=window)
        return ok, st
    if kind == "predictor":
        # forecast the error at nominal landing from the measurement and the drains
        # of the last L_nominal decisions (assumes each lands exactly L_nominal - sensor later)
        ehat = np.zeros(n + 1)
        pending = {}
        in_force = 0.0
        run, settle = 0, None
        for t in range(n):
            if t in pending:
                in_force = pending.pop(t)
            meas = e[t - sensor] if t - sensor >= 0 else e[0]
            recent = sum(g * ehat[t - i] for i in range(1, L_nominal + 1) if t - i >= 0)
            ehat[t] = meas - recent
            dt = int(lat[t])
            pending[t + dt] = g * ehat[t]
            e[t + 1] = e[t] - kappa * in_force
            if abs(e[t + 1]) > 1e6:
                return False, None
            if abs(e[t + 1]) <= band:
                run += 1
                if run >= window and settle is None:
                    settle = t + 1 - window + 1
            else:
                run, settle = 0, None
        return settle is not None, settle
    if kind == "gated":
        # pulse sized g*e over one period, applied when it lands; next decision only
        # after the pulse's effect is visible (landing + 1 + sensor)
        t, run, settle = 0, 0, None
        cur = e[0]
        steps = 0
        k = 0
        while steps < n:
            dt = int(lat[k % len(lat)])
            k += 1
            drain = g * cur
            steps += dt + 1 + sensor
            cur = cur - kappa * drain
            if abs(cur) > 1e6:
                return False, None
            if abs(cur) <= band:
                return True, steps
        return False, None
    raise ValueError(kind)


def e0c(seeds: int = 200) -> dict:
    sensor, median = 1, 2.0
    out = {"rows": []}
    for kind, g in (("every_step", 1.0), ("every_step", th.g_fastest(3)), ("predictor", 1.0), ("gated", 1.0)):
        for sigma in (0.0, 0.5, 1.0):
            for kappa in (0.5, 0.8, 1.0, 1.25, 1.6):
                oks, sts = 0, []
                for s in range(seeds):
                    r = np.random.default_rng(5000 + s)
                    lat = np.ceil(median * np.exp(r.normal(0, sigma, 800))).astype(int) if sigma else np.full(800, 2)
                    lat = np.minimum(lat, 8)
                    ok, st = simulate_construct(kind, g, sensor, lat, kappa, L_nominal=sensor + 2)
                    oks += ok
                    if ok:
                        sts.append(st)
                out["rows"].append({"construct": kind, "gain": round(g, 4), "latency_sigma": sigma, "kappa": kappa,
                                    "p_settle": oks / seeds,
                                    "median_settle_steps": float(np.median(sts)) if sts else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".local/share/stability-lab/data/e0"))
    ap.add_argument("--only", default="a,b,c")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "a" in args.only:
        r = e0a()
        (out / "e0a.json").write_text(json.dumps(r, indent=1))
        print(f"E0a: theory agrees with sim on {r['agree_theory']}/{r['n']} cells; "
              f"with the finite-horizon correction on {r['agree_theory_with_horizon']}/{r['n']}")
        for c in r["cells"]:
            if (c["sim_settled_frac"] >= 0.5) != c["observable"]:
                print("  disagreement:", c)
    if "b" in args.only:
        r = e0b()
        (out / "e0b.json").write_text(json.dumps(r, indent=1))
        for d in r["dists"]:
            print(f"E0b sigma={d['sigma']} cap={d['cap']} stats={d['latency_stats_steps']} "
                  f"g50(lww)={d['g50_last_writer_wins']} g50(guard)={d['g50_guard']} "
                  f"pred={ {k: round(v, 3) for k, v in d['predictions'].items()} }")
    if "c" in args.only:
        r = e0c()
        (out / "e0c.json").write_text(json.dumps(r, indent=1))
        for row in r["rows"]:
            print("E0c", row)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------- E0d
def simulate_sequential_agent(c_nominal: float, sensor: int, lat: np.ndarray, median: float,
                              n: int = 800, band: float = 0.05, window: int = 20):
    """Sequential agent loop on an integrating plant.

    Cycle j: observe e at t_j - sensor, think for lat[j] steps, then command a drain
    rate of e_obs / H per step, where H = median / c_nominal steps is the drain
    horizon the agent plans for (c_nominal = 1: "drain it by my next check").
    The next cycle starts as soon as the command lands, so each command stays in
    force for one full (random) thinking time.
    """
    H = median / c_nominal
    e = [0.0] * (n + 2)
    e[0] = 1.0
    t = 0
    rate = 0.0
    j = 0
    run, settle = 0, None
    obs = e[0]
    while t < n:
        d = int(lat[j % len(lat)])
        j += 1
        for _ in range(d):
            if t >= n:
                break
            e[t + 1] = e[t] - rate
            a = abs(e[t + 1])
            if a > 1e6:
                return False, None
            if a <= band:
                run += 1
                if run >= window and settle is None:
                    settle = t + 2 - window
            else:
                run, settle = 0, None
            t += 1
        # command lands now; it was decided on the observation taken at cycle start
        rate = obs / H
        obs = e[max(0, t - sensor)]
    return settle is not None, settle


def e0d(seeds: int = 60) -> dict:
    sensor, median = 1, 2.0
    rows = []
    for sigma in (0.0, 0.25, 0.5, 1.0):
        lats = []
        for s in range(seeds):
            r = np.random.default_rng(9000 + s)
            lat = np.ceil(median * np.exp(r.normal(0, sigma, 900))).astype(int) if sigma else np.full(900, int(median))
            lats.append(np.maximum(lat, 1))
        pts = []
        for c in np.round(np.arange(0.1, 1.51, 0.05), 3):
            ok = sum(simulate_sequential_agent(float(c), sensor, lat, median)[0] for lat in lats)
            pts.append((float(c), ok / seeds))
        rows.append({"sigma": sigma, "curve": pts})
        print(f"E0d sigma={sigma}: " + " ".join(f"{c:.2f}:{p:.2f}" for c, p in pts), flush=True)
    return {"sensor": sensor, "median_latency_steps": median, "rows": rows}
