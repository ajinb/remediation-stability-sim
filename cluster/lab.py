"""Real-cluster harness for the remediation-loop stability experiments.

One *slot* is one namespace running the queue-worker plant from plant.yaml.tpl.
The harness runs on the host and owns three things the paper's model names:

* the disturbance and the load (an exact-rate producer thread pushing jobs into
  Redis through a NodePort, so the arrival rate is known, not estimated);
* the sensor (queue length and completion counter, read every control period T,
  delivered to the controller ``sensor_delay`` periods late);
* the controller (a policy object) and its dead time (``dead_time`` periods
  between a decision and the scale patch landing on the API server, or the
  real wall-clock latency of an LLM call).

Everything that happens is appended to a JSONL trace; analysis lives elsewhere.
"""

from __future__ import annotations

import json
import math
import os
import socket
import string
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kubernetes import client, config

HERE = Path(__file__).resolve().parent
KUBECONFIG = os.environ.get("KUBECONFIG", str(Path.home() / ".local/share/stability-lab/kubeconfig"))


# --------------------------------------------------------------------------- redis
class Redis:
    """Minimal RESP2 client: enough for LLEN/RPUSH/INCR/GET/DEL, no dependency."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        self.sock = socket.create_connection(self.addr, timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b""

    def _readline(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _readexact(self, n: int) -> bytes:
        while len(self.buf) < n + 2:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed connection")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n + 2:]
        return data

    def _reply(self):
        line = self._readline()
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            raise RuntimeError(rest.decode())
        if kind == b":":
            return int(rest)
        if kind == b"$":
            n = int(rest)
            return None if n < 0 else self._readexact(n).decode()
        if kind == b"*":
            n = int(rest)
            return None if n < 0 else [self._reply() for _ in range(n)]
        raise RuntimeError(f"bad RESP reply {line!r}")

    def cmd(self, *args):
        parts = [b"*%d\r\n" % len(args)]
        for a in args:
            b = str(a).encode()
            parts.append(b"$%d\r\n%s\r\n" % (len(b), b))
        payload = b"".join(parts)
        with self.lock:
            for attempt in range(3):
                try:
                    self.sock.sendall(payload)
                    return self._reply()
                except (OSError, ConnectionError):
                    if attempt == 2:
                        raise
                    time.sleep(0.2)
                    self._connect()


# ---------------------------------------------------------------------- kubernetes
def kube():
    config.load_kube_config(config_file=KUBECONFIG)
    return client.AppsV1Api(), client.CoreV1Api()


def render_plant(ns: str, nodeport: int, service_time: float) -> str:
    tpl = string.Template((HERE / "plant.yaml.tpl").read_text())
    return tpl.substitute(NS=ns, NODEPORT=nodeport, SERVICE_TIME=service_time)


def apply_plant(ns: str, nodeport: int, service_time: float) -> None:
    manifest = render_plant(ns, nodeport, service_time)
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    # kubectl on this host is old, but `apply` of core objects is stable across versions.
    subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest.encode(), env=env,
                   check=True, capture_output=True)


# ------------------------------------------------------------------------ producer
class Producer(threading.Thread):
    """Pushes jobs at an exact, changeable rate using absolute-time bookkeeping."""

    def __init__(self, redis: Redis, rate: float, tick: float = 0.1):
        super().__init__(daemon=True)
        self.redis = redis
        self.rate = rate
        self.tick = tick
        self.stop_flag = threading.Event()
        self.pushed = 0
        self._owed = 0.0

    def set_rate(self, rate: float) -> None:
        self.rate = rate

    def run(self) -> None:
        last = time.monotonic()
        while not self.stop_flag.is_set():
            now = time.monotonic()
            self._owed += self.rate * (now - last)
            last = now
            n = int(self._owed)
            if n > 0:
                self._owed -= n
                self.redis.cmd("RPUSH", "jobs", *([1] * n))
                self.pushed += n
            time.sleep(self.tick)

    def stop(self) -> None:
        self.stop_flag.set()


# --------------------------------------------------------------------------- slots
@dataclass
class Slot:
    ns: str
    nodeport: int
    service_time: float = 1.0

    def __post_init__(self):
        self.apps, self.core = kube()
        self.redis: Redis | None = None

    def up(self, wait: float = 120.0) -> None:
        apply_plant(self.ns, self.nodeport, self.service_time)
        deadline = time.time() + wait
        while time.time() < deadline:
            d = self.apps.read_namespaced_deployment("redis", self.ns)
            if (d.status.ready_replicas or 0) >= 1:
                break
            time.sleep(1)
        for _ in range(60):
            try:
                self.redis = Redis("127.0.0.1", self.nodeport)
                if self.redis.cmd("PING") == "PONG":
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"redis in {self.ns} not reachable on {self.nodeport}")

    def scale(self, replicas: int) -> float:
        t0 = time.time()
        self.apps.patch_namespaced_deployment_scale(
            "worker", self.ns, {"spec": {"replicas": int(replicas)}})
        return time.time() - t0

    def worker_status(self) -> tuple[int, int, int]:
        """(spec replicas, ready replicas, running pods not terminating)."""
        d = self.apps.read_namespaced_deployment("worker", self.ns)
        pods = self.core.list_namespaced_pod(self.ns, label_selector="app=worker")
        running = sum(1 for p in pods.items
                      if p.status.phase == "Running" and p.metadata.deletion_timestamp is None)
        return d.spec.replicas or 0, d.status.ready_replicas or 0, running

    def queue(self) -> int:
        return int(self.redis.cmd("LLEN", "jobs"))

    def done(self) -> int:
        v = self.redis.cmd("GET", "stat:done")
        return int(v) if v else 0

    def reset(self) -> None:
        self.redis.cmd("DEL", "jobs")
        self.redis.cmd("SET", "stat:done", 0)

    def push(self, n: int) -> None:
        if n > 0:
            self.redis.cmd("RPUSH", "jobs", *([1] * n))

    def start_load(self, rate: float) -> "Producer":
        prod = Producer(self.redis, rate)
        prod.start()
        return prod

    def wait_workers(self, n: int, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            spec, ready, running = self.worker_status()
            if running == n and ready == n:
                return
            time.sleep(0.5)
        raise RuntimeError(f"{self.ns}: workers did not settle at {n}")


# --------------------------------------------------------------------------- clock
class RealClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    time = staticmethod(time.time)  # bound last: it shadows the module name in this body


# ------------------------------------------------------------------------ policies
@dataclass
class Obs:
    step: int           # control step the measurement was taken at
    t: float            # wall-clock time of the measurement
    queue: int
    done: int
    replicas: int       # replicas the controller last commanded (spec)
    running: int


class Policy:
    """Maps the (delayed) observation history to a replica command, or None."""

    name = "base"

    def decide(self, k: int, visible: list[Obs], ctx: "RunContext") -> int | None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RunContext:
    T: float
    arrival_rate: float
    mu: float                   # per-worker throughput the controller believes
    setpoint: int
    r_min: int
    r_max: int
    sensor_delay: int
    dead_time: int
    pending: list = field(default_factory=list)   # dicts: land, replicas[, hold_steps, restore_to]
    timers: list = field(default_factory=list)    # dicts: step, replicas (plan restores, no dead time)
    history: list = field(default_factory=list)   # all Obs, undelayed (harness only)
    commanded: dict = field(default_factory=dict)  # step -> replicas in force
    last_landed_step: int | None = None
    pulse_end_step: int | None = None

    def clamp(self, r: float) -> int:
        return int(max(self.r_min, min(self.r_max, round(r))))

    def replicas_at(self, j: int) -> int:
        """Replicas in force during step j, from history plus the known future schedule."""
        past = [s for s in self.commanded if s <= j]
        r = self.commanded[max(past)] if past else 0
        events = [(p["land"], p["replicas"]) for p in self.pending] + \
                 [(t["step"], t["replicas"]) for t in self.timers]
        for step, rr in sorted(events):
            if max(past, default=-10**9) < step <= j:
                r = rr
        return r


class Proportional(Policy):
    """Every-step proportional remediation with arrival-rate feedforward.

    r = (lambda + g * (q - q*) / T) / mu. With exact feedforward the backlog
    error obeys e[k+1] = e[k] - g * e[k-L]: the delay-integrator loop.
    """

    name = "proportional"

    def __init__(self, gain: float):
        self.gain = gain

    def decide(self, k, visible, ctx):
        if not visible:
            return None
        e = visible[-1].queue - ctx.setpoint
        return ctx.clamp((ctx.arrival_rate + self.gain * e / ctx.T) / ctx.mu)


class Gated(Proportional):
    """Dead-time-aware gating: never decide while an action is in flight or its
    effect is not yet visible.

    An action that lands (is applied to the API server) at step m first moves the
    queue measured at m + 1; the controller sees that measurement sensor_delay
    steps later. One further step is held because new pods start part-way through
    the interval in which they are created, so the first post-landing interval
    shows only part of the correction. Gating is event-based: it keys on the
    landing the harness observed, not on the nominal dead time, so it stays
    correct when the dead time is random.
    """

    name = "gated"

    def __init__(self, gain: float, extra_hold: int = 0):
        super().__init__(gain)
        self.extra_hold = extra_hold

    def decide(self, k, visible, ctx):
        if ctx.pending or not visible:
            return None
        if ctx.last_landed_step is not None and \
                k < ctx.last_landed_step + 1 + ctx.sensor_delay + 1 + self.extra_hold:
            return None
        return super().decide(k, visible, ctx)


class Cooldown(Proportional):
    """Fixed hold of `hold` steps after every landed action (classic cooling period)."""

    name = "cooldown"

    def __init__(self, gain: float, hold: int):
        super().__init__(gain)
        self.hold = hold

    def decide(self, k, visible, ctx):
        if ctx.pending or not visible:
            return None
        if ctx.last_landed_step is not None and k < ctx.last_landed_step + self.hold:
            return None
        return super().decide(k, visible, ctx)


class Predictor(Proportional):
    """Smith-predictor form: forecast the queue at the landing step from the known
    replica schedule (in force, pending, and timed restores), then apply the
    proportional law to the forecast instead of the stale measurement."""

    name = "predictor"

    def decide(self, k, visible, ctx):
        if not visible:
            return None
        obs = visible[-1]
        q = float(obs.queue)
        land = k + ctx.dead_time
        for j in range(obs.step, land):
            r_j = ctx.replicas_at(j)
            q = max(0.0, q + ctx.T * (ctx.arrival_rate - ctx.mu * r_j))
        e = q - ctx.setpoint
        return ctx.clamp((ctx.arrival_rate + self.gain * e / ctx.T) / ctx.mu)


class PulseGated(Policy):
    """Event-gated pulse remediation for an integrating plant.

    Observe the backlog error e, issue a *plan*: run r_ss + dr replicas for D
    periods, then restore r_ss (the restore is executed by the actuator side,
    not re-decided), sized so the pulse drains g*e jobs. Then hold until a
    measurement taken after the pulse ended is visible. In the ideal model the
    error at successive decisions obeys e' = (1 - g*kappa) e, whatever the delay.
    """

    name = "pulse"

    def __init__(self, gain: float, deadband: float = 0.0):
        self.gain = gain
        self.deadband = deadband
        self.blocked_until = -10**9

    def decide(self, k, visible, ctx):
        if ctx.pending or ctx.timers or not visible or k < self.blocked_until:
            return None
        if ctx.pulse_end_step is not None and visible[-1].step < ctx.pulse_end_step + 1:
            return None
        e = visible[-1].queue - ctx.setpoint
        r_ss = ctx.clamp(ctx.arrival_rate / ctx.mu)
        if abs(e) <= self.deadband:
            return None
        drain = self.gain * e                                   # jobs to remove (negative: let it refill)
        head = (ctx.r_max - r_ss) if drain > 0 else (r_ss - ctx.r_min)
        per_step_max = max(1e-9, ctx.mu * ctx.T * head)
        D = max(1, math.ceil(abs(drain) / per_step_max))
        dr = drain / (ctx.mu * ctx.T * D)
        r_pulse = ctx.clamp(r_ss + dr)
        if r_pulse == r_ss:
            return None
        return {"replicas": r_pulse, "hold_steps": D, "restore_to": r_ss}


# ------------------------------------------------------------------------- runner
@dataclass
class RunConfig:
    run_id: str
    policy: str
    gain: float = 1.0
    hold: int = 0
    T: float = 5.0
    steps: int = 70
    warmup_steps: int = 4
    arrival_rate: float = 6.0
    mu: float = 1.0
    setpoint: int = 150
    burst: int = 100
    burst_step: int = 3
    sensor_delay: int = 1
    dead_time: int = 0
    dead_time_jitter: float = 0.0   # lognormal sigma on dead time (steps)
    r_min: int = 1
    r_max: int = 30
    seed: int = 0
    notes: str = ""


def make_policy(cfg: RunConfig) -> Policy:
    if cfg.policy == "proportional":
        return Proportional(cfg.gain)
    if cfg.policy == "gated":
        return Gated(cfg.gain, extra_hold=cfg.hold)
    if cfg.policy == "cooldown":
        return Cooldown(cfg.gain, cfg.hold)
    if cfg.policy == "predictor":
        return Predictor(cfg.gain)
    if cfg.policy == "pulse":
        return PulseGated(cfg.gain)
    raise ValueError(cfg.policy)


def run(slot, cfg: RunConfig, out: Path, policy: Policy | None = None, clock=None) -> dict:
    """Execute one run against a real Slot or a twin.FakeSlot (with its VirtualClock)."""
    import random

    clock = clock or RealClock()
    rng = random.Random(cfg.seed)
    policy = policy or make_policy(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    trace = out.open("w")

    def log(rec: dict) -> None:
        trace.write(json.dumps(rec) + "\n")
        trace.flush()

    log({"type": "config", **asdict(cfg), "policy_name": policy.name, "ns": slot.ns,
         "started": clock.time()})

    # ---- reset to steady state: r_ss workers, queue at setpoint, producer at lambda
    r_ss = int(round(cfg.arrival_rate / cfg.mu))
    slot.scale(0)
    slot.wait_workers(0)
    slot.reset()
    slot.push(cfg.setpoint)
    slot.scale(r_ss)
    slot.wait_workers(r_ss)
    producer = slot.start_load(cfg.arrival_rate)

    ctx = RunContext(T=cfg.T, arrival_rate=cfg.arrival_rate, mu=cfg.mu, setpoint=cfg.setpoint,
                     r_min=cfg.r_min, r_max=cfg.r_max, sensor_delay=cfg.sensor_delay,
                     dead_time=cfg.dead_time)
    commanded = r_ss
    t0 = clock.monotonic()
    total = cfg.warmup_steps + cfg.steps
    try:
        for i in range(total):
            k = i - cfg.warmup_steps          # control step; negative during warmup
            target = t0 + i * cfg.T
            while True:
                now = clock.monotonic()
                if now >= target:
                    break
                clock.sleep(min(0.05, target - now))
            lateness = clock.monotonic() - target

            # sense
            q = slot.queue()
            dn = slot.done()
            spec, ready, running = slot.worker_status()
            obs = Obs(step=k, t=clock.time(), queue=q, done=dn, replicas=commanded, running=running)
            ctx.history.append(obs)
            ctx.commanded[k] = commanded

            # disturbance
            if k == cfg.burst_step and cfg.burst:
                slot.push(cfg.burst)

            applied = None
            decision = None
            if k >= 0:
                def actuate(r: int) -> None:
                    nonlocal commanded, applied
                    if r != commanded:
                        slot.scale(r)
                        commanded = r
                        applied = r
                        ctx.last_landed_step = k

                # plan restores fire on the actuator side, without dead time
                for tmr in [t for t in ctx.timers if t["step"] <= k]:
                    actuate(tmr["replicas"])
                ctx.timers = [t for t in ctx.timers if t["step"] > k]

                # land due actions (FIFO by landing step)
                due = sorted((p for p in ctx.pending if p["land"] <= k), key=lambda p: p["land"])
                ctx.pending = [p for p in ctx.pending if p["land"] > k]
                for p in due:
                    actuate(p["replicas"])
                    if p.get("hold_steps"):
                        ctx.timers.append({"step": k + p["hold_steps"], "replicas": p["restore_to"]})
                        ctx.pulse_end_step = k + p["hold_steps"]
                ctx.commanded[k] = commanded

                # decide on the delayed view
                visible = [o for o in ctx.history if o.step <= k - cfg.sensor_delay and o.step >= 0]
                decision = policy.decide(k, visible, ctx)
                if decision is not None:
                    plan = decision if isinstance(decision, dict) else {"replicas": decision}
                    dt = cfg.dead_time
                    if cfg.dead_time_jitter > 0 and cfg.dead_time > 0:
                        dt = max(0, round(cfg.dead_time * math.exp(rng.gauss(0, cfg.dead_time_jitter))))
                    if dt == 0:
                        actuate(plan["replicas"])
                        if plan.get("hold_steps"):
                            ctx.timers.append({"step": k + plan["hold_steps"], "replicas": plan["restore_to"]})
                            ctx.pulse_end_step = k + plan["hold_steps"]
                        ctx.commanded[k] = commanded
                    else:
                        ctx.pending.append({"land": k + dt, **plan})

            log({"type": "step", "k": k, "t": obs.t, "lateness": round(lateness, 4),
                 "queue": q, "done": dn, "spec": spec, "ready": ready, "running": running,
                 "commanded": commanded, "decision": decision, "applied": applied,
                 "pushed": producer.pushed})
    finally:
        producer.stop()
        producer.join(timeout=2)
        slot.scale(0)
        log({"type": "end", "ended": clock.time()})
        trace.close()
    return {"run_id": cfg.run_id, "trace": str(out)}
