"""Metrics for run traces (real cluster or twin): one row per run."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load(path: Path) -> tuple[dict, list[dict]]:
    cfg, steps = {}, []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if rec["type"] == "config":
            cfg = rec
        elif rec["type"] == "step":
            steps.append(rec)
    return cfg, steps


def envelope_rate(errors: list[float], floor: float) -> float | None:
    """Per-step growth factor of the |error| envelope from successive local extrema."""
    peaks = []
    for i in range(1, len(errors) - 1):
        a, b, c = abs(errors[i - 1]), abs(errors[i]), abs(errors[i + 1])
        if b >= a and b >= c and b > floor:
            if not peaks or i - peaks[-1][0] >= 2:
                peaks.append((i, b))
    if len(peaks) < 3:
        return None
    xs = [p[0] for p in peaks]
    ys = [math.log(p[1]) for p in peaks]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return math.exp(slope)


def metrics(path: Path, band: float = 20.0, window: int = 8) -> dict:
    cfg, steps = load(path)
    steps = [s for s in steps if s["k"] >= 0]
    T = cfg["T"]
    q_star = cfg["setpoint"]
    k0 = cfg["burst_step"] + 1                      # first measurement including the burst
    post = [s for s in steps if s["k"] >= k0]
    e = [s["queue"] - q_star for s in post]

    settle = None
    run = 0
    for i, s in enumerate(post):
        if abs(s["queue"] - q_star) <= band:
            run += 1
            if run >= window:
                settle = i - window + 1
                break
        else:
            run = 0
    # a run only counts as settled if it also stays settled to the end
    stays = settle is not None and all(abs(x) <= 2.5 * band for x in e[settle:])

    cmd = [s["commanded"] for s in steps]
    dirs = [1 if b > a else -1 for a, b in zip(cmd, cmd[1:]) if b != a]
    reversals = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    r_ss = round(cfg["arrival_rate"] / cfg["mu"])
    tail = e[-20:] if len(e) >= 20 else e
    return {
        "run_id": cfg["run_id"], "policy": cfg["policy"], "gain": cfg["gain"],
        "sensor_delay": cfg["sensor_delay"], "dead_time": cfg["dead_time"],
        "L": cfg["sensor_delay"] + cfg["dead_time"],
        "notes": json.loads(cfg["notes"]) if cfg.get("notes") else {},
        "settled": bool(stays), "settle_steps": settle if stays else None,
        "peak_error": max(e) if e else None, "undershoot": min(e) if e else None,
        "queue_empty_steps": sum(1 for s in post if s["queue"] == 0),
        "reversals": reversals,
        "tail_amplitude": (max(tail) - min(tail)) if tail else None,
        "envelope_rate": envelope_rate(e, floor=band),
        "pod_seconds_excess": sum(max(0, s["running"] - r_ss) for s in post) * T,
        "backlog_job_seconds_excess": sum(max(0, x) for x in e) * T,
        "max_lateness_s": max(s["lateness"] for s in steps) if steps else None,
        "steps": len(steps),
    }


def table(paths: list[Path]) -> list[dict]:
    return [metrics(p) for p in sorted(paths)]


if __name__ == "__main__":
    import sys

    rows = table([Path(p) for p in sys.argv[1:]])
    for r in rows:
        print(json.dumps(r))
