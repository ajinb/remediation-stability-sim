"""Discrete-time simulation of the Remediation Control Loop (RCL).

Plant: a bounded scalar state x (utilization-like, clamped to [x_min, x_max])
holding at a setpoint until a persistent step disturbance shifts it. Actions
are persistent shifts in the opposite direction (capacity changes), landing
after the controller's dead-time.

Controller: acts on a delayed measurement. Direction is always correct given
the measurement (the paper's premise: individually-correct decisions);
magnitude and dead-time are randomized to emulate a probabilistic controller.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class PlantConfig:
    setpoint: float = 50.0
    disturbance_at: int | None = 20     # step index of persistent disturbance; None = quiet
    disturbance: float = 40.0
    noise_sd: float = 0.5
    sensor_delay: int = 2               # measurement lags plant by this many steps
    x_min: float = 0.0
    x_max: float = 200.0


@dataclass
class ControllerConfig:
    gain: float = 0.8                   # proportional gain on measured error
    dead_time_mean: int = 1             # steps between decision and action landing (>= 1)
    dead_time_jitter: float = 0.0       # lognormal sd on dead-time
    magnitude_jitter: float = 0.0       # lognormal sd on action magnitude
    act_threshold: float = 5.0          # |error| must exceed this to act
    period: int = 1                     # controller runs every `period` steps


@dataclass
class Constructs:
    """Stable-by-design constructs from §5. All off = greedy baseline."""

    hysteresis: bool = False            # §5.1: engage/release bands
    engage_band: float = 8.0
    release_band: float = 3.0
    damping: bool = False               # §5.2: cap per-action magnitude
    max_step: float = 20.0
    cooldown: bool = False              # §5.3: hold after an action lands
    settle_steps: int = 10
    deadtime_aware: bool = False        # §5.4: at most one action in flight
    interaction_matrix: bool = False    # §5.5: actuator mutex across loops


@dataclass
class Result:
    trajectory: list[float]
    actions: int
    reversals: int
    osc_amplitude: float
    overshoot: float
    settling_time: int | None
    per_controller_actions: list[int] = field(default_factory=list)

    # Settling time doubles as time-to-repair for a repairing disturbance.
    @property
    def mttr(self) -> int | None:
        return self.settling_time


class _Controller:
    def __init__(self, cfg: ControllerConfig, constructs: Constructs, rng: random.Random):
        self.cfg = cfg
        self.constructs = constructs
        self.rng = rng
        self.in_flight: list[tuple[int, float]] = []  # (land_step, delta)
        self.hold_until = -1
        self.lock_until = -1                # actuator lock: issue → land + settle
        self.last_direction = 0
        self.issued = 0

    def holds_lock(self, t: int) -> bool:
        """§5.5: a loop owns the actuator while an action is in flight *and*
        through its settling window — otherwise peers act on telemetry that
        does not yet show the correction."""
        return bool(self.in_flight) or t < self.lock_until

    def _dead_time(self) -> int:
        mean = max(1, self.cfg.dead_time_mean)
        if self.cfg.dead_time_jitter <= 0:
            return mean
        return max(1, round(mean * math.exp(self.rng.gauss(0.0, self.cfg.dead_time_jitter))))

    def decide(self, t: int, measurement: float, setpoint: float, actuator_busy: bool) -> None:
        c = self.constructs
        if t % self.cfg.period != 0:
            return
        if c.interaction_matrix and actuator_busy:
            return                      # §5.5: another loop owns the actuator
        if c.deadtime_aware and self.in_flight:
            return                      # §5.4: never act blind on your own dead-time
        if c.cooldown and t < self.hold_until:
            return                      # §5.3: wait out the settling window

        error = measurement - setpoint
        direction = -1 if error > 0 else 1
        if c.hysteresis:
            # §5.1: fresh engagements and direction reversals must cross the
            # engage band; once engaged, keep correcting in the same direction
            # until the error falls inside the release band.
            continuing = self.last_direction != 0 and direction == self.last_direction
            threshold = c.release_band if continuing else c.engage_band
        else:
            threshold = self.cfg.act_threshold
        if abs(error) <= threshold:
            return

        magnitude = self.cfg.gain * abs(error)
        if self.cfg.magnitude_jitter > 0:
            magnitude *= math.exp(self.rng.gauss(0.0, self.cfg.magnitude_jitter))
        if c.damping:
            magnitude = min(magnitude, c.max_step)  # §5.2

        land = t + self._dead_time()
        self.in_flight.append((land, direction * magnitude))
        self.lock_until = land + c.settle_steps
        if c.cooldown:
            self.hold_until = land + c.settle_steps
        self.last_direction = direction
        self.issued += 1

    def landing(self, t: int) -> float:
        due = [d for (land, d) in self.in_flight if land == t]
        self.in_flight = [(land, d) for (land, d) in self.in_flight if land > t]
        return sum(due)


def _metrics(trajectory, landed_dirs, plant: PlantConfig, actions: int,
             per_controller: list[int], settle_band: float, settle_window: int) -> Result:
    reversals = sum(1 for a, b in zip(landed_dirs, landed_dirs[1:]) if a != b)

    start = plant.disturbance_at if plant.disturbance_at is not None else 0
    post = trajectory[start:]
    osc_amplitude = (max(post) - min(post)) if post else 0.0
    # Overshoot: how far the loop drove the plant past the setpoint on the
    # side opposite the disturbance.
    if plant.disturbance_at is not None and plant.disturbance > 0:
        overshoot = max(0.0, plant.setpoint - min(post))
    elif plant.disturbance_at is not None and plant.disturbance < 0:
        overshoot = max(0.0, max(post) - plant.setpoint)
    else:
        overshoot = 0.0

    settling_time = None
    run = 0
    for i, x in enumerate(trajectory[start:], start=0):
        if abs(x - plant.setpoint) <= settle_band:
            run += 1
            if run >= settle_window:
                settling_time = i - settle_window + 1
                break
        else:
            run = 0
    return Result(trajectory=trajectory, actions=actions, reversals=reversals,
                  osc_amplitude=osc_amplitude, overshoot=overshoot,
                  settling_time=settling_time, per_controller_actions=per_controller)


def run_multi_scenario(plant: PlantConfig, controllers: list[ControllerConfig],
                       constructs: Constructs, steps: int = 400, seed: int = 0,
                       settle_band: float = 5.0, settle_window: int = 20) -> Result:
    """Run one scenario with one or more controllers sharing the actuator."""
    rng = random.Random(seed)
    ctrls = [_Controller(cfg, constructs, rng) for cfg in controllers]

    # Persistent level moved by disturbances and landed actions; observed x
    # adds transient noise on top (fluctuation, not a random walk).
    level = plant.setpoint
    trajectory: list[float] = []
    landed_dirs: list[int] = []

    for t in range(steps):
        if plant.disturbance_at is not None and t == plant.disturbance_at:
            level += plant.disturbance

        for ctrl in ctrls:
            delta = ctrl.landing(t)
            if delta:
                level += delta
                landed_dirs.append(1 if delta > 0 else -1)

        level = min(plant.x_max, max(plant.x_min, level))
        x = min(plant.x_max, max(plant.x_min, level + rng.gauss(0.0, plant.noise_sd)))
        trajectory.append(x)

        measurement = trajectory[max(0, t - plant.sensor_delay)]
        for ctrl in ctrls:
            # Busy is evaluated live so the mutex also covers two loops
            # deciding within the same step (list order = lock priority).
            busy = any(o is not ctrl and o.holds_lock(t) for o in ctrls)
            ctrl.decide(t, measurement, plant.setpoint, actuator_busy=busy)

    return _metrics(trajectory, landed_dirs, plant, sum(c.issued for c in ctrls),
                    [c.issued for c in ctrls], settle_band, settle_window)


def run_scenario(plant: PlantConfig, controller: ControllerConfig,
                 constructs: Constructs, steps: int = 400, seed: int = 0,
                 settle_band: float = 5.0, settle_window: int = 20) -> Result:
    return run_multi_scenario(plant, [controller], constructs, steps=steps, seed=seed,
                              settle_band=settle_band, settle_window=settle_window)
