"""Render the paper's data figures as vector PDFs from the study outputs.

    python paper/make_figures.py --data ~/.local/share/stability-lab/data --out <dir> [--only fig2,fig5]

Palette: the first slots of a validated categorical palette (colour-vision-deficiency
separation checked), always paired with a line style or marker and a direct label so
no series is identified by colour alone; figures stay legible in greyscale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "cluster"))
import theory as th  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad5"
MM = 1 / 25.4

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK, va="bottom")


# ------------------------------------------------------------------------------ fig 2
def fig_stability_map(data: Path, out: Path) -> None:
    e0a = json.loads((data / "e0" / "e0a.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 72 * MM), gridspec_kw={"width_ratios": [1.35, 1]})

    Ls = [int(L) for L in range(0, 21)]
    Lf = np.linspace(0, 20, 400)
    gstar = np.array([th.g_star(L) for L in Ls])
    a.fill_between(Ls, gstar, 3.0, color="#f0efec", zorder=0, lw=0)
    a.plot(Ls, gstar, color=C[0], marker="o", ms=2.6, lw=1.4, zorder=3)
    a.plot(Lf, math.pi / (2 * (Lf + 1)), color=C[1], ls="--", lw=1.2, zorder=3)
    a.plot(Ls[1:], [th.varying_delay_gain(K) for K in Ls[1:]], color=C[2], ls="-.", lw=1.2, zorder=3)
    Lopt = [L for L in Ls[1:] if th.g_fastest(L) >= 0.033]
    a.plot(Lopt, [th.g_fastest(L) for L in Lopt], color=C[6], ls=":", lw=1.4, zorder=3)
    for c in e0a["cells"]:
        settled = c["sim_settled_frac"] >= 0.5
        a.scatter(c["L"], c["gain"], s=11, marker="s", zorder=4,
                  facecolor=INK if settled else "white", edgecolor=INK, linewidth=0.6)
    a.axhline(1.0, color=INK2, lw=0.6, ls=(0, (1, 2)))
    a.text(Lopt[-1] + 0.3, th.g_fastest(Lopt[-1]), "fastest-settling gain", color=C[6], fontsize=6, va="center")
    a.text(19.8, 1.9, "unstable: spectral radius > 1", color=INK2, fontsize=6.5, style="italic", ha="right")
    a.text(19.8, 1.06, "full correction, $g=1$", ha="right", va="bottom", color=INK2, fontsize=6)
    a.set_yscale("log")
    a.set_ylim(0.03, 2.6)
    a.set_xlim(-0.3, 20.6)
    a.set_xticks(range(0, 21, 2))
    a.set_xlabel("loop delay $L$ (control periods)")
    a.set_ylabel("correction gain $g$ per period")
    a.grid(True, which="major", axis="both")
    a.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    a.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C[0], marker="o", ms=2.6), Line2D([], [], color=C[1], ls="--"),
         Line2D([], [], color=C[2], ls="-."), Line2D([], [], color=C[6], ls=":"),
         Line2D([], [], marker="s", ls="", mfc=INK, mec=INK, ms=3.5),
         Line2D([], [], marker="s", ls="", mfc="white", mec=INK, ms=3.5)]
    lab = ["exact boundary", "product rule (sufficient)", "3/2 bound, varying delay", "fastest-settling gain",
           "prior frontier: settled", "prior frontier: not settled"]
    a.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
             handlelength=1.8, columnspacing=1.2)
    panel_label(a, "a")

    # (b) repair-time floor
    Lb = np.arange(1, 21)
    floor20 = [th.repair_floor_steps(int(L), 20.0) for L in Lb]
    b.plot(Lb, floor20, color=C[0], marker="o", ms=2.8, label="every-step control, best gain")
    b.plot(Lb, Lb * math.log(20), color=INK2, ls=":", lw=0.9, label="lower bound $L\\ln 20$")
    b.plot(Lb, Lb + 2, color=C[1], ls="--", marker="^", ms=2.8, label="event-gated pulse, one cycle")
    b.set_xlabel("loop delay $L$ (control periods)")
    b.set_ylabel("periods to shrink the error 20-fold")
    b.grid(True)
    b.set_xlim(0.5, 20.5)
    b.set_xticks(range(2, 21, 2))
    b.set_ylim(0, 70)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    panel_label(b, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig2_stability_map.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------ fig 5
def fig_latency_architecture(data: Path, out: Path) -> None:
    e0b = json.loads((data / "e0" / "e0b.json").read_text())
    e0d = json.loads((data / "e0" / "e0d.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 58 * MM), sharey=True)

    styles = {0.25: ("-", "o"), 0.5: ("--", "s"), 1.0: ("-.", "^")}
    for i, d in enumerate(dd for dd in e0b["dists"] if dd["cap"] is None):
        pts = d["curves"]["guard"]
        g = [p[0] for p in pts]
        p = [p[1] for p in pts]
        ls, mk = styles[d["sigma"]]
        a.plot(g, p, color=C[i], ls=ls, marker=mk, ms=2.6, label=f"latency spread $\\sigma={d['sigma']}$")
    a.axvline(th.g_star(3), color=INK2, lw=0.7, ls=(0, (1, 2)))
    a.text(th.g_star(3) + 0.01, 0.06, "$g^\\star$ at the\nmedian delay", color=INK2, fontsize=6, va="bottom")
    a.set_xlabel("correction gain $g$ per period")
    a.set_ylabel("share of runs that settle")
    a.set_title("overlapping decisions (one issued every period)", loc="left", color=INK)
    a.grid(True)
    a.legend(loc="lower left", frameon=False)
    panel_label(a, "a")

    styles_d = {0.0: ("-", "o"), 0.25: ("--", "s"), 0.5: ("-.", "^"), 1.0: (":", "D")}
    for i, row in enumerate(e0d["rows"]):
        c = [p[0] for p in row["curve"]]
        p = [p[1] for p in row["curve"]]
        ls, mk = styles_d[row["sigma"]]
        b.plot(c, p, color=C[i], ls=ls, marker=mk, ms=2.6, label=f"$\\sigma={row['sigma']}$")
    b.set_xlabel("planned aggressiveness (median think time / drain horizon)")
    b.set_title("sequential agent loop (observe, think, act)", loc="left", color=INK)
    b.grid(True)
    b.legend(loc="lower left", frameon=False)
    panel_label(b, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig5_latency_architecture.pdf")
    plt.close(fig)


FIGS = {"fig2": fig_stability_map, "fig5": fig_latency_architecture}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / ".local/share/stability-lab/data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(FIGS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        FIGS[name](Path(args.data), out)
        print("wrote", name)



# ------------------------------------------------------------------------------ fig 3
def _trace(path: Path):
    import analyze
    cfg, steps = analyze.load(path)
    steps = [s for s in steps if s["k"] >= 0]
    k = np.array([s["k"] for s in steps])
    e = np.array([s["queue"] - cfg["setpoint"] for s in steps])
    r = np.array([s["commanded"] for s in steps])
    return cfg, k, e, r


def fig_cluster_frontier(data: Path, out: Path) -> None:
    import analyze
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 64 * MM), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) measured envelope rate against predicted spectral radius
    marks = {1: "o", 2: "s", 3: "^", 5: "D"}
    for p in sorted((data / "e1").glob("*.jsonl")):
        m = analyze.metrics(p)
        if m["envelope_rate"] is None:
            continue
        rho = th.spectral_radius(m["gain"], m["L"], 0.3, 1.0)
        a.scatter(rho, m["envelope_rate"], marker=marks[m["L"]], s=18, linewidth=0.8, zorder=3,
                  facecolor=C[0] if m["settled"] else "white", edgecolor=C[0] if m["settled"] else C[1])
    lim = (0.85, 1.2)
    a.plot(lim, lim, color=INK2, lw=0.7, ls=(0, (2, 2)))
    a.axvline(1.0, color=INK2, lw=0.6)
    a.axhline(1.0, color=INK2, lw=0.6)
    a.text(1.195, 1.03, "saturation caps growth near 1.0", color=INK2, fontsize=6, ha="right", va="bottom")
    a.text(0.875, 0.885, "identity", color=INK2, fontsize=6, rotation=38)
    from matplotlib.ticker import FormatStrFormatter
    a.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    a.set_xlim(*lim)
    a.set_ylim(0.85, 1.06)
    a.set_yticks([0.85, 0.90, 0.95, 1.00, 1.05])
    a.set_xlabel("predicted spectral radius (nominal plant)")
    a.set_ylabel("measured envelope rate per period")
    a.grid(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=marks[L], ls="", mfc="none", mec=INK2, ms=4) for L in (1, 2, 3, 5)]
    h += [Line2D([], [], marker="o", ls="", mfc=C[0], mec=C[0], ms=4), Line2D([], [], marker="o", ls="", mfc="white", mec=C[1], ms=4)]
    a.legend(h, ["$L=1$", "$L=2$", "$L=3$", "$L=5$", "settled", "did not settle"], loc="lower right",
             frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)
    panel_label(a, "a")

    # (b) trajectories at L = 3 either side of the limit, cluster and twin
    for margin, col in ((0.85, C[0]), (1.15, C[1])):
        for rep in (0, 1):
            cfg, k, e, r = _trace(data / "e1" / f"e1-L3-m{margin}-r{rep}.jsonl")
            b.plot(k * cfg["T"], e, color=col, lw=1.2 if rep == 0 else 0.7, alpha=1.0 if rep == 0 else 0.55)
        cfg, k, e, r = _trace(data / "twin" / f"e1-L3-m{margin}-r0.jsonl")
        b.plot(k * cfg["T"], e, color=col, lw=0.9, ls=(0, (3, 2)))
    b.axhspan(-20, 20, color="#f0efec", zorder=0, lw=0)
    b.text(348, 112, "1.15 of the limit: limit cycle", color=C[1], fontsize=6.5, ha="right", va="top")
    b.text(348, -118, "0.85 of the limit: settles", color=C[0], fontsize=6.5, ha="right", va="top")
    b.text(348, -140, "solid: cluster runs; dashed: digital twin", color=INK2, fontsize=6, ha="right", va="top")
    b.set_xlabel("time since start of control (s)")
    b.set_ylabel("backlog above target (jobs)")
    b.set_ylim(-160, 120)
    b.set_xlim(0, 350)
    b.grid(True)
    panel_label(b, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig3_cluster_frontier.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------ fig 4
def fig_constructs(data: Path, out: Path) -> None:
    order = [("baseline-g1", "every-step, $g=1$"), ("cooldown-g1-h4", "cooldown of 4 periods, $g=1$"),
             ("margin-half", "gain at half the limit"), ("fastest-gopt", "fastest-settling gain"),
             ("predictor-g1", "predictor, $g=1$"), ("pulse-gated-g1", "event-gated pulse, $g=1$")]
    fig, axes = plt.subplots(2, 3, figsize=(170 * MM, 88 * MM), sharex=True, sharey=True)
    for ax, (name, title) in zip(axes.ravel(), order):
        ax.axhspan(-20, 20, color="#f0efec", zorder=0, lw=0)
        for rep in (0, 1):
            cfg, k, e, r = _trace(data / "e2" / f"e2-{name}-r{rep}.jsonl")
            ax.plot(k * cfg["T"], e, color=C[0], lw=1.1 if rep == 0 else 0.7, alpha=1.0 if rep == 0 else 0.6)
        cfg, k, e, r = _trace(data / "twin" / f"e2-{name}-r0.jsonl")
        ax.plot(k * cfg["T"], e, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", color=INK, fontsize=7)
        ax.grid(True)
        ax.set_ylim(-160, 130)
        ax.set_xlim(0, 350)
        ax.set_xticks([0, 100, 200, 300])
    for ax in axes[1]:
        ax.set_xlabel("time since start of control (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("backlog above target (jobs)")
    axes[0, 2].text(345, -95, "solid: two cluster runs\ndashed: digital twin\nshaded: settle band", color=INK2,
                    fontsize=6, ha="right", va="top")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.savefig(out / "fig4_constructs.pdf")
    plt.close(fig)


FIGS.update({"fig3": fig_cluster_frontier, "fig4": fig_constructs})


# ------------------------------------------------------------------------------ fig 6
def fig_llm_probe(data: Path, out: Path) -> None:
    rows = []
    for sub in ("haiku", "sonnet"):
        p = data / "e3a" / sub / "probe.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    rows = [r for r in rows if r["ok"]]
    models = [("claude-haiku-4-5", "Haiku 4.5", C[0]), ("claude-sonnet-5", "Sonnet 5", C[1])]
    rng = np.random.default_rng(3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), gridspec_kw={"width_ratios": [1, 1.6]})
    for i, (mid, name, col) in enumerate(models):
        lat = np.array([r["wall_s"] for r in rows if r["model"] == mid])
        y = i + rng.uniform(-0.18, 0.18, len(lat))
        a.scatter(lat, y, s=10, color=col, alpha=0.85, linewidth=0, zorder=3)
        med = float(np.median(lat))
        a.plot([med, med], [i - 0.32, i + 0.32], color=INK, lw=1.4, zorder=4)
        a.text(med, i + 0.36, f"median {med:.0f} s", ha="center", va="bottom", fontsize=6, color=INK)
    a.set_yticks([0, 1])
    a.set_yticklabels([m[1] for m in models])
    a.set_ylim(-0.6, 1.75)
    a.set_xscale("log")
    a.set_xticks([10, 20, 50, 100])
    a.set_xticklabels(["10", "20", "50", "100"])
    a.set_xlabel("decision latency per call (s)")
    a.grid(True, axis="x")
    panel_label(a, "a")

    groups = []
    for mid, name, col in models:
        for T in (5, 30):
            for pend in (False, True):
                groups.append((mid, name, col, T, pend))
    x = 0
    ticks, labels = [], []
    pair_x = []
    for mid, name, col, T, pend in groups:
        sel = [r for r in rows if r["model"] == mid and r["input"]["T"] == T and bool(r["input"].get("pending")) == pend]
        H = np.array([T / r["g_eff"] for r in sel if r["g_eff"] > 0])
        xs = x + rng.uniform(-0.2, 0.2, len(H))
        b.scatter(xs, H, s=10, marker="o" if not pend else "^", facecolor=col if not pend else "white",
                  edgecolor=col, linewidth=0.8, zorder=3)
        lat = np.median([r["wall_s"] for r in rows if r["model"] == mid])
        hmin = 2 / math.pi * (lat + 2 + T)
        b.plot([x - 0.35, x + 0.35], [hmin, hmin], color=INK, lw=1.2, zorder=4)
        pair_x.append(x)
        if pend:
            ticks.append((pair_x[-2] + pair_x[-1]) / 2)
            labels.append(f"{T} s")
        x += 1 if not pend else 1.6
    b.set_yscale("log")
    b.set_yticks([5, 10, 20, 50, 100, 200])
    b.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    b.set_xticks(ticks)
    b.set_xticklabels(labels, fontsize=6.5)
    b.set_ylabel("planned drain horizon $H$ (s)")
    b.set_xlabel("next invocation in")
    b.grid(True, axis="y")
    b.text((ticks[0] + ticks[1]) / 2, 230, "Haiku 4.5", color=C[0], ha="center", fontsize=6.5)
    b.text((ticks[2] + ticks[3]) / 2, 230, "Sonnet 5", color=C[1], ha="center", fontsize=6.5)
    b.set_ylim(3, 300)
    from matplotlib.lines import Line2D
    b.legend([Line2D([], [], color=INK, lw=1.2), Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4),
              Line2D([], [], marker="^", ls="", mfc="white", mec=INK2, ms=4)],
             ["minimum stable horizon (product rule)", "no scale-up in flight", "scale-up already in flight"],
             loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)
    panel_label(b, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig6_llm_probe.pdf")
    plt.close(fig)


FIGS.update({"fig6": fig_llm_probe})


# ------------------------------------------------------------------------------ fig 7
def fig_closed_loop(data: Path, out: Path) -> None:
    models = [("claude-haiku-4-5", "Haiku 4.5"), ("claude-sonnet-5", "Sonnet 5")]
    conds = [("stateless", "re-invoked as soon as it acts"), ("pulse", "event-gated pulse")]
    fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 92 * MM), sharex=True, sharey=True)
    for i, (mid, mname) in enumerate(models):
        for j, (cond, cname) in enumerate(conds):
            ax = axes[i, j]
            ax.axhspan(-20, 20, color="#f0efec", zorder=0, lw=0)
            for rep, (col, lw) in enumerate(((C[0], 1.2), (C[1], 1.0))):
                path = data / "e3b" / f"e3b-{mid}-{cond}-r{rep}.jsonl"
                if not path.exists():
                    continue
                recs = [json.loads(l) for l in path.read_text().splitlines()]
                cfg = recs[0]
                t0 = next(r for r in recs if r["type"] == "burst")["t"]
                smp = [r for r in recs if r["type"] == "sample" and -12 <= r["t"] - t0 <= cfg["duration"]]
                ax.plot([r["t"] - t0 for r in smp], [r["queue"] - cfg["target"] for r in smp], color=col, lw=lw,
                        label=f"run {rep + 1}")
                dec = [r for r in recs if r["type"] == "decision" and r.get("ok")]
                for d in dec:
                    ax.plot([d["t_return"] - t0] * 2, [-150 + 8 * rep, -138 + 8 * rep], color=col, lw=1.0)
            ax.set_title(f"{mname}, {cname}", loc="left", color=INK, fontsize=7)
            ax.grid(True)
            ax.set_ylim(-155, 115)
            ax.set_xlim(-12, 300)
    for ax in axes[1]:
        ax.set_xlabel("time since the 100-job burst (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("backlog above target (jobs)")
    axes[0, 1].legend(loc="upper right", frameon=False)
    axes[0, 0].text(295, -112, "ticks: decisions applied", color=INK2, fontsize=6, ha="right")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    fig.savefig(out / "fig7_closed_loop.pdf")
    plt.close(fig)


FIGS.update({"fig7": fig_closed_loop})


if __name__ == "__main__":
    main()
