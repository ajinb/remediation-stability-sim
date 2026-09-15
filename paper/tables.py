"""Build the paper's result tables (LaTeX rows + JSON) directly from run traces.

    python paper/tables.py --data ~/.local/share/stability-lab/data --out <dir>

No number in the results tables is transcribed by hand; each comes from here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "cluster"))
import analyze  # noqa: E402
import identify  # noqa: E402
import theory as th  # noqa: E402

HORIZON = 70 - 4        # control periods from the first post-burst measurement to the end of a run


def fmt(x, nd=2, dash="--"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return dash
    return f"{x:.{nd}f}"


def scale_ops(path: Path) -> int:
    _, steps = analyze.load(path)
    cmd = [s["commanded"] for s in steps if s["k"] >= 0]
    return sum(1 for a, b in zip(cmd, cmd[1:]) if a != b)


def predicted_class(g: float, L: int, phi: float = 0.3, kappa: float = 1.0) -> tuple[str, float]:
    rho = th.spectral_radius(g, L, phi, kappa)
    if rho >= 1:
        return "unstable", rho
    need = math.log(100 / 20) / math.log(1 / rho)    # envelope 100 jobs -> settle band 20 jobs
    return ("settles" if need <= HORIZON else "slow"), rho


def e1_table(data: Path) -> dict:
    runs = sorted((data / "e1").glob("*.jsonl"))
    twin = {p.stem: analyze.metrics(p) for p in (data / "twin").glob("e1-*.jsonl")}
    cells = defaultdict(list)
    for p in runs:
        m = analyze.metrics(p)
        cells[(m["L"], m["notes"]["margin"])].append((p, m))
    rows = []
    agree_nominal = agree_twin = total = 0
    for (L, margin), items in sorted(cells.items()):
        g = items[0][1]["gain"]
        cls, rho = predicted_class(g, L)
        settled = [m["settled"] for _, m in items]
        pred_settles = cls == "settles"
        agree_nominal += sum(s == pred_settles for s in settled)
        total += len(settled)
        tw = twin.get(items[0][1]["run_id"].rsplit("-r", 1)[0] + "-r0")
        if tw is not None:
            agree_twin += sum(s == tw["settled"] for s in settled)
        envs = [m["envelope_rate"] for _, m in items if m["envelope_rate"] is not None]
        sts = [m["settle_steps"] for _, m in items if m["settle_steps"] is not None]
        rows.append({
            "L": L, "margin": margin, "gain": g, "rho_pred": rho, "prediction": cls,
            "settled": f"{sum(settled)}/{len(settled)}",
            "settle_periods": st.mean(sts) if sts else None,
            "envelope_rate": st.mean(envs) if envs else None,
            "undershoot": st.mean(m["undershoot"] for _, m in items),
            "empty": st.mean(m["queue_empty_steps"] for _, m in items),
            "twin_settled": None if tw is None else tw["settled"],
            "twin_settle_periods": None if tw is None else tw["settle_steps"],
        })
    return {"rows": rows, "agree_nominal": agree_nominal, "agree_twin": agree_twin, "total": total}


def e2_table(data: Path) -> dict:
    by = defaultdict(list)
    for p in sorted((data / "e2").glob("*.jsonl")):
        m = analyze.metrics(p)
        name = m["run_id"].split("-", 1)[1].rsplit("-r", 1)[0]
        by[name].append((p, m))
    twin = {p.stem.split("-", 1)[1].rsplit("-r", 1)[0]: analyze.metrics(p) for p in (data / "twin").glob("e2-*.jsonl")}
    order = ["baseline-g1", "cooldown-g1-h4", "margin-half", "fastest-gopt", "predictor-g1", "pulse-gated-g1"]
    rows = []
    for name in order:
        items = by[name]
        sts = [m["settle_steps"] for _, m in items if m["settle_steps"] is not None]
        tw = twin.get(name)
        rows.append({
            "construct": name, "gain": items[0][1]["gain"],
            "settled": f"{sum(m['settled'] for _, m in items)}/{len(items)}",
            "settle_periods": st.mean(sts) if sts else None,
            "undershoot": st.mean(m["undershoot"] for _, m in items),
            "empty": st.mean(m["queue_empty_steps"] for _, m in items),
            "excess_pod_s": st.mean(m["pod_seconds_excess"] for _, m in items),
            "scale_ops": st.mean(scale_ops(p) for p, _ in items),
            "twin_settle_periods": None if tw is None else tw["settle_steps"],
            "twin_settled": None if tw is None else tw["settled"],
        })
    return {"rows": rows}


def ident_summary(data: Path) -> dict:
    paths = [str(p) for p in sorted((data / "e1").glob("*.jsonl")) + sorted((data / "e2").glob("*.jsonl"))]
    Xs, ys, per = [], [], []
    for p in paths:
        cfg, X, y = identify.design(Path(p))
        if len(y) < 8:
            continue
        Xs.append(X)
        ys.append(y)
        f = identify.fit(X, y, cfg["T"])
        per.append({"run_id": cfg["run_id"], **f, "kappa": f["mu_eff"] / cfg["mu"]})
    pooled = identify.fit(np.vstack(Xs), np.concatenate(ys), 5.0)
    kap = [r["kappa"] for r in per]
    return {"pooled": {**pooled, "kappa": pooled["mu_eff"] / 0.97}, "kappa_range": [min(kap), max(kap)],
            "runs": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / ".local/share/stability-lab/data"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"e1": e1_table(data), "e2": e2_table(data), "identification": ident_summary(data)}
    (out / "tables.json").write_text(json.dumps(res, indent=1))

    lines = []
    for r in res["e1"]["rows"]:
        lines.append(f"{r['L']} & {r['margin']:.2f} & {r['gain']:.3f} & {r['rho_pred']:.3f} & {r['prediction']} & "
                     f"{r['settled']} & {fmt(r['settle_periods'], 1)} & {fmt(r['envelope_rate'], 3)} & "
                     f"{r['undershoot']:.0f} & {'settles' if r['twin_settled'] else 'does not'} \\\\")
    (out / "table_e1_rows.tex").write_text("\n".join(lines) + "\n")
    names = {"baseline-g1": "Every-step, full correction", "cooldown-g1-h4": "Cooldown, 4 periods",
             "margin-half": "Gain at half the limit", "fastest-gopt": "Fastest-settling gain",
             "predictor-g1": "Predictor", "pulse-gated-g1": "Event-gated pulse"}
    lines = []
    for r in res["e2"]["rows"]:
        lines.append(f"{names[r['construct']]} & {r['gain']:.4g} & {r['settled']} & {fmt(r['settle_periods'], 1)} & "
                     f"{r['undershoot']:.0f} & {r['empty']:.1f} & {r['excess_pod_s']:.0f} & {r['scale_ops']:.1f} & "
                     f"{fmt(r['twin_settle_periods'], 0) if r['twin_settled'] else 'no'} \\\\")
    (out / "table_e2_rows.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"e1_agree_nominal": res["e1"]["agree_nominal"], "e1_agree_twin": res["e1"]["agree_twin"],
                      "e1_total": res["e1"]["total"], "ident": res["identification"]["pooled"],
                      "kappa_range": res["identification"]["kappa_range"]}, indent=1))



# ------------------------------------------------------------------------- E0 tables
def aoi_mean(sigma: float, median: float = 2.0, cap=None, amax: int = 200) -> float:
    """Mean age of the command in force, Eq. (aoi), for ceil(lognormal) latencies."""
    from scipy.stats import norm

    def p_gt(j):
        if cap is not None and j >= cap:
            return 0.0
        if j < 1:
            return 1.0
        return 1.0 - norm.cdf(math.log(j / median) / sigma)
    surv, p = [], 1.0
    for a in range(amax):
        p *= p_gt(a)
        surv.append(p)
    return float(sum(surv))            # E[A] = sum_{a>=0} P(A > a)


def g_star_interp(L: float) -> float:
    lo = int(math.floor(L))
    return th.g_star(lo) + (L - lo) * (th.g_star(lo + 1) - th.g_star(lo))


def e0_tables(data: Path) -> dict:
    e0b = json.loads((data / "e0" / "e0b.json").read_text())
    e0c = json.loads((data / "e0" / "e0c.json").read_text())
    e0d = json.loads((data / "e0" / "e0d.json").read_text())
    s = e0b["sensor"]
    rows_b = []
    for d in e0b["dists"]:
        sig, cap = d["sigma"], d["cap"]
        med = e0b["median_latency_steps"]
        p95 = med * math.exp(1.6449 * sig)
        if cap is not None:
            p95 = min(p95, cap)
        ea = aoi_mean(sig, med, cap)
        rows_b.append({"sigma": sig, "cap": cap, "p95": p95, "mean_age": ea,
                       "g50_guard": d["g50_guard"], "g50_lww": d["g50_last_writer_wins"],
                       "pred_median": th.g_star(s + math.ceil(med)), "pred_p95": th.g_star(s + math.ceil(p95)),
                       "pred_age": g_star_interp(s + ea),
                       "bound_cap": th.varying_delay_gain(cap + s) if cap is not None else None})

    def c50(curve):
        for (c0, p0), (c1, p1) in zip(curve, curve[1:]):
            if p0 >= 0.5 > p1:
                return c0 + (p0 - 0.5) * (c1 - c0) / (p0 - p1)
        return None
    rows_d = [{"sigma": r["sigma"], "c50": c50(r["curve"])} for r in e0d["rows"]]
    rows_c = defaultdict(dict)
    for r in e0c["rows"]:
        key = (r["construct"], r["gain"])
        rows_c[key].setdefault(r["latency_sigma"], []).append(r)
    summary_c = []
    for (construct, gain), by_sigma in rows_c.items():
        for sig, rs in sorted(by_sigma.items()):
            ps = [x["p_settle"] for x in rs]
            meds = [x["median_settle_steps"] for x in rs if x["median_settle_steps"] is not None]
            summary_c.append({"construct": construct, "gain": gain, "sigma": sig,
                              "p_settle_min": min(ps), "p_settle_max": max(ps),
                              "per_kappa": {x["kappa"]: x["p_settle"] for x in rs},
                              "median_settle_range": [min(meds), max(meds)] if meds else None})
    return {"e0b": rows_b, "e0c": summary_c, "e0d": rows_d}


# ------------------------------------------------------------------------- E3a table
def e3a_table(data: Path) -> dict:
    rows = []
    for sub in ("haiku", "sonnet"):
        p = data / "e3a" / sub / "probe.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines()]

    def q(xs, frac):
        xs = sorted(xs)
        i = (len(xs) - 1) * frac
        lo = int(math.floor(i))
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    out = []
    for model in ("claude-haiku-4-5", "claude-sonnet-5"):
        ok = [r for r in rows if r["model"] == model and r["ok"]]
        if not ok:
            continue
        lat = [r["wall_s"] for r in ok]
        med_lat = st.median(lat)
        for T in (5, 30):
            for pend in (False, True):
                sel = [r for r in ok if r["input"]["T"] == T and bool(r["input"].get("pending")) == pend]
                g = [r["g_eff"] for r in sel]
                H = [T / x for x in g if x > 0]
                h_min = 2 / math.pi * (med_lat + 2 + T)          # Eq. (horizon), D = latency + start-up
                out.append({"model": model, "T": T, "pending": pend, "n": len(sel),
                            "latency_median": med_lat, "latency_p90": q(lat, 0.9), "latency_max": max(lat),
                            "api_median": st.median(r["api_ms"] / 1000 for r in ok),
                            "out_tokens_median": st.median(r["usage"]["output_tokens"] for r in ok),
                            "g_median": st.median(g), "g_min": min(g), "g_max": max(g),
                            "H_median": st.median(H), "H_min_required": h_min,
                            "stacked": sum(1 for r in sel if pend and r["replicas"] > r["input"]["pending"]),
                            "cut_back": sum(1 for r in sel if pend and r["replicas"] < r["input"]["pending"]),
                            "calls_failed": sum(1 for r in rows if r["model"] == model and not r["ok"])})
    return {"rows": out, "n_calls": len(rows)}


def write_generated(data: Path, out: Path) -> None:
    e0 = e0_tables(data)
    (out / "e0_tables.json").write_text(json.dumps(e0, indent=1))
    lines = []
    for r in e0["e0b"]:
        cap = "none" if r["cap"] is None else str(r["cap"])
        lines.append(f"{r['sigma']:.2f} & {cap} & {r['p95']:.1f} & {r['mean_age']:.2f} & {r['g50_guard']:.3f} & "
                     f"{r['pred_median']:.3f} & {r['pred_p95']:.3f} & {r['pred_age']:.3f} \\\\")
    (out / "table_e0b_rows.tex").write_text("\n".join(lines) + "\n")
    e3a = e3a_table(data)
    (out / "e3a_table.json").write_text(json.dumps(e3a, indent=1))
    names = {"claude-haiku-4-5": "Haiku 4.5", "claude-sonnet-5": "Sonnet 5"}
    lines = []
    for r in e3a["rows"]:
        cond = "in flight" if r["pending"] else "none"
        extra = f"{r['stacked']} / {r['cut_back']}" if r["pending"] else "--"
        lines.append(f"{names[r['model']]} & {r['T']} & {cond} & {r['g_median']:.2f} [{r['g_min']:.2f}, {r['g_max']:.2f}] & "
                     f"{r['H_median']:.0f} & {r['H_min_required']:.0f} & {extra} \\\\")
    (out / "table_e3a_rows.tex").write_text("\n".join(lines) + "\n")



# ------------------------------------------------------------------------- E3b table
def e3b_metrics(path: Path, band: float = 20.0, hold_s: float = 40.0) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    cfg = recs[0]
    burst = next(r for r in recs if r["type"] == "burst")
    t0 = burst["t"]
    target, r_ss = cfg["target"], round(cfg["lam"] / cfg["mu"])
    samples = [r for r in recs if r["type"] == "sample" and 0 <= r["t"] - t0 <= cfg["duration"]]
    decisions = [r for r in recs if r["type"] == "decision"]
    e = [(r["t"] - t0, r["queue"] - target) for r in samples]
    settle = None
    start = None
    for t, x in e:
        if abs(x) <= band:
            start = t if start is None else start
            if t - start >= hold_s and settle is None:
                settle = start
        else:
            start = None
            if settle is not None:
                settle = None
    stays = settle is not None and all(abs(x) <= 2.5 * band for t, x in e if t >= settle)
    specs = [r["spec"] for r in samples]
    ops = sum(1 for a, b in zip(specs, specs[1:]) if a != b)
    excess = sum(max(0, r["running"] - r_ss) for r in samples) * 1.0
    lat = [d["wall_s"] for d in decisions if d.get("ok")]
    return {"run_id": cfg["run_id"], "model": cfg["model"], "condition": cfg["condition"],
            "decisions": len(decisions), "failed": sum(1 for d in decisions if not d.get("ok")),
            "latency_median": st.median(lat) if lat else None,
            "settle_s": settle if stays else None, "settled": bool(stays),
            "min_error": min(x for _, x in e), "max_error": max(x for _, x in e),
            "empty_s": sum(1 for r in samples if r["queue"] == 0),
            "excess_pod_s": excess, "scale_ops": ops,
            "tail_amplitude": (lambda tail: max(tail) - min(tail))([x for t, x in e if t >= cfg["duration"] - 100]),
            "aborted": any(r["type"] == "abort" for r in recs)}


def e3b_table(data: Path) -> dict:
    rows = [e3b_metrics(p) for p in sorted((data / "e3b").glob("*.jsonl"))]
    return {"rows": rows}


if __name__ == "__main__":
    main()
    import sys as _sys
    _args = _sys.argv
    _out = Path(_args[_args.index("--out") + 1]) if "--out" in _args else None
    if _out is not None:
        _data = (Path(_args[_args.index("--data") + 1]) if "--data" in _args
                 else Path.home() / ".local/share/stability-lab/data")
        write_generated(_data, _out)
        e3b = e3b_table(_data)
        (_out / "e3b_table.json").write_text(json.dumps(e3b, indent=1))
        _names = {"claude-haiku-4-5": "Haiku 4.5", "claude-sonnet-5": "Sonnet 5"}
        _lines = []
        for r in sorted(e3b["rows"], key=lambda r: (r["model"], r["condition"], r["run_id"])):
            _lines.append(f"{_names[r['model']]} & {r['condition']} & {int(r['run_id'][-1]) + 1} & {r['decisions']} & "
                          f"{fmt(r['latency_median'], 1)} & {'yes' if r['settled'] else 'no'} & "
                          f"{fmt(r['settle_s'], 0)} & {r['min_error']:.0f} & "
                          f"{r['excess_pod_s']:.0f} & {r['scale_ops']} \\\\")
        (_out / "table_e3b_rows.tex").write_text("\n".join(_lines) + "\n")
