"""E3b: an LLM agent in the loop on the real queue-worker plant.

A sequential agent loop — observe, call the model, apply its decision, observe
again — runs against one plant slot while a producer holds the arrival rate and a
burst of jobs is injected. Two platform conditions:

  stateless  the agent sets a replica count and is re-invoked as soon as it has
             been applied (the common "act, then look again" loop);
  pulse      the platform executes the decision as a temporary change (the agent
             picks the replica count and how long to hold it; the platform then
             restores the steady-state count on its own) and re-invokes the agent
             only after the hold has ended and a further 5 s observation window has
             passed: the event-gated pulse of the paper's Proposition 4.

A 1 Hz sensor thread records the plant independently of the agent.

    python closed_loop.py --model claude-haiku-4-5 --condition stateless --slot 3 --rep 0
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cluster"))
import claude_cli  # noqa: E402
from lab import Slot  # noqa: E402

SLOTS = {1: ("lab-1", 31001), 2: ("lab-2", 31002), 3: ("lab-3", 31003)}
DATA = Path.home() / ".local/share/stability-lab/data/e3b"

SYSTEM = ("You are an autonomous site-reliability agent. You control the replica count of one "
          "Kubernetes Deployment of queue workers. Keep the queue backlog near its target without "
          "wasting capacity. Respond with the requested JSON only.")
SCHEMA = {"type": "object",
          "properties": {"replicas": {"type": "integer"},
                         "rationale": {"type": "string", "maxLength": 300}},
          "required": ["replicas", "rationale"], "additionalProperties": False}
PULSE_SCHEMA = {"type": "object",
                "properties": {"replicas": {"type": "integer"},
                               "hold_seconds": {"type": "integer"},
                               "rationale": {"type": "string", "maxLength": 300}},
                "required": ["replicas", "hold_seconds", "rationale"], "additionalProperties": False}


class Sensor(threading.Thread):
    def __init__(self, slot: Slot, log, period: float = 1.0):
        super().__init__(daemon=True)
        self.slot, self.log, self.period = slot, log, period
        self.stop_flag = threading.Event()
        self.samples: list[dict] = []
        self.lock = threading.Lock()

    def run(self) -> None:
        t0 = time.monotonic()
        i = 0
        while not self.stop_flag.is_set():
            target = t0 + i * self.period
            while time.monotonic() < target:
                time.sleep(0.01)
            spec, ready, running = self.slot.worker_status()
            rec = {"type": "sample", "t": time.time(), "queue": self.slot.queue(), "done": self.slot.done(),
                   "spec": spec, "running": running}
            with self.lock:
                self.samples.append(rec)
            self.log(rec)
            i += 1

    def latest(self) -> dict:
        with self.lock:
            return dict(self.samples[-1])

    def rate(self, window: float = 10.0) -> float:
        with self.lock:
            s = self.samples[-1]
            old = [x for x in self.samples if x["t"] <= s["t"] - window]
            ref = old[-1] if old else self.samples[0]
        dt = s["t"] - ref["t"]
        return (s["done"] - ref["done"]) / dt if dt > 0 else 0.0


def prompt(obs: dict, cond: str, lam: float, mu: float, target: int, r_min: int, r_max: int, r_ss: int) -> str:
    lines = [
        "Service: batch-worker (Kubernetes Deployment `worker`)",
        f"Queue backlog now: {obs['queue']} jobs (target backlog: {target} jobs)",
        f"Arrival rate: {lam:.1f} jobs/s",
        f"Each worker completes about {mu:.2f} jobs/s",
        f"Replicas: spec {obs['spec']}, running {obs['running']} (allowed range {r_min}-{r_max})",
        "New workers start processing about 2 s after a scale-up.",
    ]
    if cond == "stateless":
        lines.append("You will be called again as soon as your decision has been applied.")
        lines.append("How many replicas should the Deployment run now?")
    else:
        lines.append(f"The platform applies your decision as a temporary change: it sets the replica count you "
                     f"choose, holds it for the number of seconds you choose (5-300), then returns the "
                     f"Deployment to the steady-state count of {r_ss} replicas by itself. You are called again "
                     f"only after the hold has ended and a further 5 s of metrics are available.")
        lines.append("Which replica count should be held, and for how many seconds?")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", choices=["stateless", "pulse"], required=True)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--duration", type=float, default=300.0, help="seconds after the burst")
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()

    lam, mu, target, burst, r_min, r_max = 6.0, 0.97, 150, 100, 1, 45
    r_ss = round(lam / mu)
    ns, port = SLOTS[args.slot]
    slot = Slot(ns, port)
    slot.up()
    DATA.mkdir(parents=True, exist_ok=True)
    run_id = f"e3b-{args.model}-{args.condition}-r{args.rep}"
    path = DATA / f"{run_id}.jsonl"
    fh = path.open("w")
    lock = threading.Lock()

    def log(rec):
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    log({"type": "config", "run_id": run_id, "model": args.model, "condition": args.condition,
         "effort": args.effort, "lam": lam, "mu": mu, "target": target, "burst": burst,
         "r_min": r_min, "r_max": r_max, "duration": args.duration, "ns": ns, "started": time.time()})

    slot.scale(0)
    slot.wait_workers(0)
    slot.reset()
    slot.push(target)
    slot.scale(r_ss)
    slot.wait_workers(r_ss)
    producer = slot.start_load(lam)
    sensor = Sensor(slot, log)
    sensor.start()
    time.sleep(10)
    t_burst = time.time()
    slot.push(burst)
    log({"type": "burst", "t": t_burst, "jobs": burst})
    time.sleep(1.5)          # let the 1 Hz sensor record the burst before the first decision
    calls = 0
    try:
        while time.time() - t_burst < args.duration:
            obs = sensor.latest()
            p = prompt(obs, args.condition, lam, mu, target, r_min, r_max, r_ss)
            schema = SCHEMA if args.condition == "stateless" else PULSE_SCHEMA
            res = claude_cli.call(SYSTEM, p, schema, model=args.model, effort=args.effort)
            calls += 1
            rec = {"type": "decision", "t_obs": obs["t"], "obs": obs, "t_return": time.time(),
                   **{k: res.get(k) for k in ("ok", "wall_s", "api_ms", "usage", "error", "api_error_status")}}
            if res.get("ok"):
                r = max(r_min, min(r_max, int(res["structured"]["replicas"])))
                rec["replicas"] = r
                rec["rationale"] = res["structured"].get("rationale")
                slot.scale(r)
                rec["t_applied"] = time.time()
                if args.condition == "pulse":
                    hold = max(5, min(300, int(res["structured"].get("hold_seconds", 5))))
                    rec["hold_seconds"] = hold
            log(rec)
            if not res.get("ok") and res.get("api_error_status") in (401, 403, 429):
                log({"type": "abort", "reason": res.get("error")})
                break
            if args.condition == "pulse" and res.get("ok"):
                end_hold = rec["t_applied"] + rec["hold_seconds"]
                while time.time() < end_hold and time.time() - t_burst < args.duration:
                    time.sleep(0.2)
                slot.scale(r_ss)
                log({"type": "restore", "t": time.time(), "replicas": r_ss})
                deadline = time.time() + 60
                while time.time() < deadline:
                    s_ = sensor.latest()
                    if s_["running"] == s_["spec"] and s_["t"] > time.time() - 1.5:
                        break
                    time.sleep(0.5)
                time.sleep(5.0)
    finally:
        producer.stop()
        sensor.stop_flag.set()
        sensor.join(timeout=3)
        slot.scale(0)
        log({"type": "end", "t": time.time(), "calls": calls})
        fh.close()
    print(f"{run_id}: {calls} calls")


if __name__ == "__main__":
    main()
