"""Identify the real plant from run traces and test the loop model against it.

For each run, the queue balance over one control period is

    q[k+1] - q[k] - lambda*T - burst[k] = -mu_eff*T * ((1 - phi) c[k] + phi c[k-1]) + noise

where c[k] is the replica count commanded at step k. New pods start part-way
through the period, so a share phi of each period still runs at the previous
count. Least squares on (c[k], c[k-1]) gives mu_eff and phi; steps whose queue
touches empty are excluded (the balance does not hold when workers idle).

With the identified plant gain kappa = mu_eff / mu_hat and phi, the loop model
e[k+1] = e[k] - g*kappa((1-phi) e[k-L] + phi e[k-L-1]) predicts a spectral radius
per run, compared against the measured envelope rate of the backlog error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
import analyze  # noqa: E402
import theory as th  # noqa: E402


def design(path: Path, min_queue: int = 15):
    cfg, steps = analyze.load(path)
    steps = [s for s in steps if s["k"] >= 0]
    T, lam = cfg["T"], cfg["arrival_rate"]
    X, y = [], []
    for i in range(1, len(steps) - 1):
        s0, s1, sp = steps[i], steps[i + 1], steps[i - 1]
        if min(s0["queue"], s1["queue"]) < min_queue:
            continue
        if s0["lateness"] > 1.0 or s1["lateness"] > 1.0:
            continue
        burst = cfg["burst"] if s0["k"] == cfg["burst_step"] else 0
        # pushed counts the producer's exact arrivals over the period
        arrivals = s1["pushed"] - s0["pushed"]
        X.append([s0["commanded"], sp["commanded"]])
        y.append(s1["queue"] - s0["queue"] - arrivals - burst)
    return cfg, np.array(X, float), np.array(y, float)


def fit(X: np.ndarray, y: np.ndarray, T: float) -> dict:
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    mu_eff = -(a + b) / T
    phi = b / (a + b) if (a + b) != 0 else float("nan")
    pred = X @ coef
    ss = float(np.sum((y - pred) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"mu_eff": float(mu_eff), "phi": float(phi), "r2": 1 - ss / tot, "n": int(len(y))}


def main(paths: list[str], out: str | None = None) -> None:
    rows, Xall, yall = [], [], []
    for p in sorted(paths):
        cfg, X, y = design(Path(p))
        if len(y) < 8:
            continue
        f = fit(X, y, cfg["T"])
        m = analyze.metrics(Path(p))
        kappa = f["mu_eff"] / cfg["mu"]
        L = cfg["sensor_delay"] + cfg["dead_time"]
        row = {"run_id": cfg["run_id"], "policy": cfg["policy"], "gain": cfg["gain"], "L": L, **f,
               "kappa": kappa, "settled": m["settled"], "envelope_rate": m["envelope_rate"],
               "tail_amplitude": m["tail_amplitude"], "queue_empty_steps": m["queue_empty_steps"],
               "max_lateness_s": m["max_lateness_s"]}
        if cfg["policy"] == "proportional":
            row["rho_pred_nominal"] = th.spectral_radius(cfg["gain"], L, 0.3, 1.0)
            if 0 <= f["phi"] < 1:
                row["rho_pred_identified"] = th.spectral_radius(cfg["gain"], L, f["phi"], kappa)
        rows.append(row)
        Xall.append(X)
        yall.append(y)
    pooled = fit(np.vstack(Xall), np.concatenate(yall), 5.0)
    result = {"pooled": {**pooled, "kappa": pooled["mu_eff"] / 0.97}, "runs": rows}
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    print("pooled:", json.dumps(result["pooled"]))
    for r in rows:
        print(f"{r['run_id']:24s} g={r['gain']:<6} L={r['L']} mu_eff={r['mu_eff']:.3f} phi={r['phi']:.2f} "
              f"kappa={r['kappa']:.2f} r2={r['r2']:.2f} n={r['n']:2d} settled={r['settled']!s:5} "
              f"env={r['envelope_rate'] if r['envelope_rate'] is None else round(r['envelope_rate'], 3)} "
              f"rho_nom={r.get('rho_pred_nominal', float('nan')):.3f} rho_id={r.get('rho_pred_identified', float('nan')):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:], out=None)
