"""Experiment grids for the real-cluster study, sharded across plant slots.

    python experiments.py smoke  --slot 1
    python experiments.py e1     --slot 1 --shard 0/2 [--reps 2]
    python experiments.py e2     --slot 2 --shard 1/2

Each run writes <out>/<experiment>/<run_id>.jsonl; completed runs are skipped
on restart, so a shard can be re-launched after an interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import RunConfig, Slot, run  # noqa: E402

DATA = Path.home() / ".local/share/stability-lab/data"
SLOTS = {1: ("lab-1", 31001), 2: ("lab-2", 31002), 3: ("lab-3", 31003)}

# Plant constants from calibration (cluster/calibrate.py, 2026-09-14):
# mu ~ 0.97 jobs/s per worker; scale-up lands ~1.8 s after the patch (phi ~ 0.36
# of a 5 s period), scale-down ~1.0 s (phi ~ 0.2).
BASE = RunConfig(run_id="", policy="proportional", T=5.0, steps=70, warmup_steps=4,
                 arrival_rate=6.0, mu=0.97, setpoint=150, burst=100, burst_step=3,
                 sensor_delay=1, dead_time=0, r_min=1, r_max=45)


def gcrit(L: int, phi: float) -> float:
    """Stability boundary of e[k+1] = e[k] - g(1-phi) e[k-L] - g phi e[k-L-1]."""
    import numpy as np

    def rho(g: float) -> float:
        n = L + 2
        c = np.zeros(n + 1)
        c[0], c[1] = 1.0, -1.0
        c[n - 1] += g * (1 - phi)
        c[n] += g * phi
        return float(max(abs(np.roots(c))))

    lo, hi = 1e-6, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rho(mid) < 1 else (lo, mid)
    return lo


def grid_smoke() -> list[RunConfig]:
    return [replace(BASE, run_id="smoke-prop-g0.3-L1", gain=0.3, steps=20)]


def grid_e1(reps: int) -> list[RunConfig]:
    """Frontier validation: every-step proportional control across L and margin."""
    runs = []
    phi = 0.3
    for dead_time in (0, 1, 2, 4):
        L = BASE.sensor_delay + dead_time
        g_star = gcrit(L, phi)
        for margin in (0.6, 0.85, 1.15, 1.5):
            g = round(margin * g_star, 3)
            for rep in range(reps):
                runs.append(replace(BASE, run_id=f"e1-L{L}-m{margin}-r{rep}", gain=g,
                                    dead_time=dead_time, seed=rep,
                                    notes=json.dumps({"L": L, "margin": margin, "g_star_phi0.3": g_star})))
    return runs


def grid_e2(reps: int) -> list[RunConfig]:
    """Constructs at a naive operating point: full correction (g = 1) with L = 3."""
    dead_time = 2
    L = BASE.sensor_delay + dead_time
    g_star = gcrit(L, 0.3)
    g_opt = L ** L / (L + 1) ** (L + 1)
    configs = [
        ("baseline-g1", dict(policy="proportional", gain=1.0)),
        ("margin-half", dict(policy="proportional", gain=round(0.5 * g_star, 3))),
        ("fastest-gopt", dict(policy="proportional", gain=round(g_opt, 4))),
        ("predictor-g1", dict(policy="predictor", gain=1.0)),
        ("pulse-gated-g1", dict(policy="pulse", gain=1.0)),
        ("cooldown-g1-h4", dict(policy="cooldown", gain=1.0, hold=4)),
    ]
    runs = []
    for name, kw in configs:
        for rep in range(reps):
            runs.append(replace(BASE, run_id=f"e2-{name}-r{rep}", dead_time=dead_time, seed=rep,
                                notes=json.dumps({"L": L, "g_star_phi0.3": g_star, "g_opt": g_opt}), **kw))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["smoke", "e1", "e2"])
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    grid = {"smoke": grid_smoke, "e1": lambda: grid_e1(args.reps), "e2": lambda: grid_e2(args.reps)}[args.experiment]()
    i, n = (int(x) for x in args.shard.split("/"))
    mine = [cfg for j, cfg in enumerate(grid) if j % n == i]
    ns, port = SLOTS[args.slot]
    slot = Slot(ns, port)
    slot.up()
    out_dir = DATA / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.experiment} shard {i}/{n} on {ns}: {len(mine)} runs", flush=True)
    for cfg in mine:
        path = out_dir / f"{cfg.run_id}.jsonl"
        if path.exists() and '"type": "end"' in path.read_text():
            print("skip", cfg.run_id, flush=True)
            continue
        t0 = time.time()
        run(slot, cfg, path)
        print(f"done {cfg.run_id} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
