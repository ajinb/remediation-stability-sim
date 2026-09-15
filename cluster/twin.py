"""Virtual-time digital twin of the queue-worker plant.

Implements the same interface lab.run() drives on a real Slot, so the identical
controller loop, policies and trace format run in milliseconds. Pod start and
stop lags, per-worker throughput and its jitter come from cluster calibration;
the twin is used to pre-register predictions for the cluster grids and to test
the policies, and the twin-versus-cluster gap is itself reported.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


class VirtualClock:
    def __init__(self, plant: "FakeSlot", dt: float = 0.05):
        self.t = 0.0
        self.plant = plant
        self.dt = dt
        plant.clock = self

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return 1.7e9 + self.t

    def sleep(self, seconds: float) -> None:
        end = self.t + max(0.0, seconds)
        while end - self.t > 1e-9:
            step = min(self.dt, end - self.t)
            self.plant.advance(step)
            self.t += step
        # always land exactly on `end`: a sub-nanosecond remainder must still
        # advance time, or a caller waiting for a deadline spins forever
        self.t = max(self.t, end)


@dataclass
class _Load:
    rate: float
    pushed: int = 0
    owed: float = 0.0

    def stop(self) -> None:
        self.rate = 0.0

    def join(self, timeout: float | None = None) -> None:
        return None


@dataclass
class FakeSlot:
    ns: str = "twin"
    mu: float = 0.97               # jobs per worker-second
    mu_jitter: float = 0.0         # per-pod lognormal sigma on throughput
    start_lag: float = 1.8         # seconds from scale-up patch to consuming
    stop_lag: float = 1.0          # seconds from scale-down patch to not consuming
    seed: int = 0
    clock: VirtualClock | None = None
    queue_f: float = 0.0
    done_f: float = 0.0
    pods: list = field(default_factory=list)   # dicts: active_at, stop_at, mu
    target: int = 0
    load: _Load | None = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    # --- interface used by lab.run -------------------------------------------------
    def up(self) -> None:
        return None

    def now(self) -> float:
        return self.clock.t if self.clock else 0.0

    def scale(self, replicas: int) -> float:
        replicas = int(replicas)
        live = [p for p in self.pods if p["stop_at"] is None]
        if replicas > len(live):
            for _ in range(replicas - len(live)):
                m = self.mu * (self.rng.lognormvariate(0, self.mu_jitter) if self.mu_jitter else 1.0)
                self.pods.append({"active_at": self.now() + self.start_lag, "stop_at": None, "mu": m})
        elif replicas < len(live):
            # the ReplicaSet controller removes not-yet-running and newest pods first
            victims = sorted(live, key=lambda p: -p["active_at"])[: len(live) - replicas]
            for p in victims:
                p["stop_at"] = self.now() + (0.0 if p["active_at"] > self.now() else self.stop_lag)
        self.target = replicas
        return 0.0

    def worker_status(self) -> tuple[int, int, int]:
        t = self.now()
        running = sum(1 for p in self.pods if p["active_at"] <= t and (p["stop_at"] is None))
        return self.target, running, running

    def queue(self) -> int:
        return int(self.queue_f)

    def done(self) -> int:
        return int(self.done_f)

    def reset(self) -> None:
        self.queue_f = 0.0
        self.done_f = 0.0

    def push(self, n: int) -> None:
        self.queue_f += n

    def start_load(self, rate: float) -> _Load:
        self.load = _Load(rate)
        return self.load

    def wait_workers(self, n: int, timeout: float = 0) -> None:
        # jump pod states to settled at n workers (used only during run setup)
        t = self.now()
        self.pods = [p for p in self.pods if p["stop_at"] is None][:n]
        for p in self.pods:
            p["active_at"] = min(p["active_at"], t)
        while len(self.pods) < n:
            self.pods.append({"active_at": t, "stop_at": None, "mu": self.mu})
        self.target = n

    # --- dynamics --------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        t = self.now()
        if self.load and self.load.rate > 0:
            self.load.owed += self.load.rate * dt
            n = int(self.load.owed)
            if n:
                self.load.owed -= n
                self.load.pushed += n
                self.queue_f += n
        cap = 0.0
        for p in self.pods:
            if p["active_at"] <= t and (p["stop_at"] is None or p["stop_at"] > t):
                cap += p["mu"]
        served = min(self.queue_f, cap * dt)
        self.queue_f -= served
        self.done_f += served
        self.pods = [p for p in self.pods if p["stop_at"] is None or p["stop_at"] > t]
