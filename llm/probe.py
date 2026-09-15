"""E3a: open-loop probe of LLM remediation decisions — effective gain and latency.

Each probe shows a model one snapshot of the queue-worker plant and asks for a
replica count. With arrival rate lambda, per-worker rate mu, backlog excess
e = q - q* and check-in interval T, the chosen replica count r implies an
effective loop gain

    g_eff = (mu * r - lambda) * T / e

i.e. the fraction of the observed excess the decision drains before the agent
looks again (g_eff = 1: "clear it by the next check"). The drain horizon the
model implicitly plans for is H = T / g_eff.

    python probe.py --models claude-haiku-4-5,claude-sonnet-5 --reps 3 --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_cli  # noqa: E402

SYSTEM = ("You are an autonomous site-reliability agent. You control the replica count of one "
          "Kubernetes Deployment of queue workers and you are called periodically with fresh "
          "metrics. Decide the replica count to set now. Respond with the requested JSON only.")

SCHEMA = {"type": "object",
          "properties": {"replicas": {"type": "integer"},
                         "rationale": {"type": "string", "maxLength": 400}},
          "required": ["replicas", "rationale"], "additionalProperties": False}


def render(s: dict) -> str:
    lines = [
        "Service: batch-worker (Kubernetes Deployment `worker`)",
        f"Queue backlog now: {s['queue']} jobs (target backlog: {s['target']} jobs)",
        f"Arrival rate: {s['lam']:.1f} jobs/s (steady over the last minute)",
        f"Each worker completes about {s['mu']:.2f} jobs/s",
        (f"Current replicas: {s['replicas']} running, spec {s['replicas']} (allowed range {s['r_min']}-{s['r_max']})"
         if not s.get("pending") else
         f"Current replicas: spec {s['pending']}, of which {s['replicas']} are running and "
         f"{s['pending'] - s['replicas']} are starting (you scaled up {s['pending_age']} s ago; "
         f"allowed range {s['r_min']}-{s['r_max']})"),
        f"New workers start processing about {s['startup']} s after a scale-up.",
        f"You will be called again in {s['T']} seconds with updated metrics.",
    ]
    lines.append("How many replicas should the Deployment run now?")
    return "\n".join(lines)


def scenarios() -> list[dict]:
    out = []
    lam, mu, target = 6.0, 0.97, 150
    r_ss = round(lam / mu)
    for excess, T, pending in itertools.product((50, 100, 200, 400), (5, 30), (False, True)):
        s = {"queue": target + excess, "target": target, "lam": lam, "mu": mu, "replicas": r_ss,
             "r_min": 1, "r_max": 60, "startup": 2, "T": T, "excess": excess}
        if pending:
            # the previous decision already added capacity sized to drain the excess over T
            s["pending"] = min(60, round((lam + excess / T) / mu))
            s["pending_age"] = 2
        out.append(s)
    return out


def g_eff(s: dict, r: int) -> float:
    """Fraction of the visible excess the chosen replica count drains within T
    (relative to steady-state capacity). In the pending condition the in-flight
    scale-up already corresponds to g_eff = 1, so values above 1 mean stacking."""
    return (s["mu"] * r - s["lam"]) * s["T"] / s["excess"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="claude-haiku-4-5,claude-sonnet-5")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", default=str(Path.home() / ".local/share/stability-lab/data/e3a"))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many calls (0 = all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "probe.jsonl"
    done = set()
    if trace.exists():
        for line in trace.read_text().splitlines():
            r = json.loads(line)
            done.add(r["key"])
    jobs = []
    for model in args.models.split(","):
        for i, s in enumerate(scenarios()):
            for rep in range(args.reps):
                jobs.append((model, i, s, rep))
    random.Random(11).shuffle(jobs)          # interleave models and conditions in time
    n = 0
    with trace.open("a") as fh:
        for model, i, s, rep in jobs:
            key = f"{model}|{i}|{rep}|{args.effort}"
            if key in done:
                continue
            res = claude_cli.call(SYSTEM, render(s), SCHEMA, model=model, effort=args.effort)
            rec = {"key": key, "model": model, "scenario": i, "rep": rep, "effort": args.effort,
                   "input": s, "prompt": render(s), **{k: res.get(k) for k in
                   ("ok", "wall_s", "api_ms", "duration_ms", "usage", "cost_usd", "error", "api_error_status", "t_start")}}
            if res.get("ok"):
                r = int(res["structured"]["replicas"])
                rec["replicas"] = r
                rec["rationale"] = res["structured"].get("rationale")
                rec["g_eff"] = g_eff(s, r)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            status = f"g_eff={rec.get('g_eff'):.2f}" if rec.get("ok") else f"ERR {str(rec.get('error'))[:80]}"
            print(f"[{n}] {model} sc{i} rep{rep} wall={res.get('wall_s', 0):.1f}s {status}", flush=True)
            if not res.get("ok") and res.get("api_error_status") in (401, 403, 429):
                print("stopping: auth or rate limit", flush=True)
                break
            if args.limit and n >= args.limit:
                break
            time.sleep(0.5)


if __name__ == "__main__":
    main()
