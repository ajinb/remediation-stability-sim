"""Calibrate the queue-worker plant: per-worker throughput and actuation lag.

Measures, with a backlog large enough that workers never idle:
  * mu: completions per worker-second at a fixed replica count;
  * scale-up lag: time from the scale patch until completions reach the new rate;
  * scale-down lag: time from the patch until completions fall to the new rate.

Writes a JSON summary and the raw 0.25 s samples.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import Slot  # noqa: E402


def sample(slot: Slot, seconds: float, period: float = 0.25) -> list[dict]:
    out = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        t = time.time()
        out.append({"t": t, "done": slot.done(), "queue": slot.queue(),
                    "running": slot.worker_status()[2]})
        time.sleep(max(0.0, period - (time.time() - t)))
    return out


def rate(samples: list[dict], t_from: float, t_to: float) -> float:
    win = [s for s in samples if t_from <= s["t"] <= t_to]
    if len(win) < 2:
        return float("nan")
    return (win[-1]["done"] - win[0]["done"]) / (win[-1]["t"] - win[0]["t"])


def lag(samples: list[dict], t_patch: float, old_rate: float, new_rate: float, window: float = 2.0) -> float:
    """First time after the patch at which a trailing `window`-second completion
    rate has covered 90% of the way from old_rate to new_rate."""
    goal = old_rate + 0.9 * (new_rate - old_rate)
    for s in samples:
        if s["t"] < t_patch + window:
            continue
        r = rate(samples, s["t"] - window, s["t"])
        if (new_rate > old_rate and r >= goal) or (new_rate < old_rate and r <= goal):
            return s["t"] - window / 2 - t_patch
    return float("nan")


def main(ns: str = "lab-1", nodeport: int = 31001, out_dir: str = "calibration") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slot = Slot(ns, nodeport)
    slot.up()
    slot.scale(0)
    slot.wait_workers(0)
    slot.reset()
    slot.redis.cmd("RPUSH", "jobs", *([1] * 3000))

    summary = {}
    # steady throughput at 10 workers
    slot.scale(10)
    slot.wait_workers(10)
    time.sleep(3)
    s1 = sample(slot, 45)
    r10 = rate(s1, s1[0]["t"] + 5, s1[-1]["t"])
    summary["mu_per_worker_at_10"] = r10 / 10

    # scale-up 10 -> 25
    t_up = time.time()
    slot.scale(25)
    s2 = sample(slot, 30)
    r25 = rate(s2, t_up + 15, s2[-1]["t"])
    summary["mu_per_worker_at_25"] = r25 / 25
    summary["scale_up_lag_s"] = lag(s1[-20:] + s2, t_up, r10, r25)
    first_running = next((s["t"] - t_up for s in s2 if s["running"] >= 25), float("nan"))
    summary["scale_up_all_running_s"] = first_running

    # scale-down 25 -> 8
    t_dn = time.time()
    slot.scale(8)
    s3 = sample(slot, 30)
    r8 = rate(s3, t_dn + 15, s3[-1]["t"])
    summary["mu_per_worker_at_8"] = r8 / 8
    summary["scale_down_lag_s"] = lag(s2[-20:] + s3, t_dn, r25, r8)

    slot.scale(0)
    summary["samples"] = {"steady10": len(s1), "up": len(s2), "down": len(s3)}
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    (out / "calibration_samples.json").write_text(json.dumps({"steady10": s1, "up": s2, "down": s3,
                                                              "t_up": t_up, "t_dn": t_dn}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:2] or ["lab-1"])
