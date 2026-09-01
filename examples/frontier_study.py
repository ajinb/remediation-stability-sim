"""E4: the empirical stability frontier of the greedy remediation loop (§6, v0.3).

The paper's §5 conditions are informal. This study makes one of them concrete:
a greedy proportional loop keeps issuing corrections while its own actions are
in flight, so over one full loop delay L = dead_time + sensor_delay it stacks
roughly gain x L worth of correction against the error it can see. The classic
prediction is a product law — the loop settles iff gain x L stays below a
constant — and stochasticity in the controller should spend some of that
margin.

The study measures the frontier directly: for each (gain, dead_time) cell,
20 seeds of the greedy loop (no constructs), settled fraction, and the
critical dead-time tau*(gain) = the largest dead-time in the sweep such that
the loop settles in >= 90% of seeds at that dead-time and every shorter one.
Swept at three controller-stochasticity levels; jitter 0.3 is the paper's E1
controller.

Usage: python examples/frontier_study.py [--seeds 20]
"""

from __future__ import annotations

import argparse

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)
SENSOR_DELAY = SPIKE["sensor_delay"]

GAINS = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9)
DEAD_TIMES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
JITTERS = (0.0, 0.3, 0.6)  # applied to both magnitude and dead-time
SETTLE_THRESHOLD = 0.9


def settle_fraction(gain: float, dead_time: int, jitter: float, seeds) -> float:
    settled = 0
    for s in seeds:
        cfg = ControllerConfig(gain=gain, dead_time_mean=dead_time,
                               dead_time_jitter=jitter, magnitude_jitter=jitter,
                               act_threshold=5.0)
        r = run_scenario(plant=PlantConfig(**SPIKE), controller=cfg,
                         constructs=Constructs(), steps=400, seed=s)
        settled += r.settling_time is not None
    return settled / len(seeds)


def critical_dead_time(gain: float, jitter: float, seeds) -> int:
    """Largest dead-time with settle fraction >= threshold at it and every
    shorter one (contiguity guards against isolated lucky cells)."""
    tau_star = 0
    for dt in DEAD_TIMES:
        if settle_fraction(gain, dt, jitter, seeds) >= SETTLE_THRESHOLD:
            tau_star = dt
        else:
            break
    return tau_star


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    seeds = range(args.seeds)

    print("=== E4: settled fraction, greedy loop, gain x dead_time "
          f"({args.seeds} seeds/cell; jitter = paper's E1 controller at 0.3) ===")
    for jitter in JITTERS:
        print(f"\n--- controller jitter {jitter} (magnitude and dead-time) ---")
        header = f"{'gain':>6} | " + "".join(f"{dt:>6}" for dt in DEAD_TIMES)
        print(header + f" | tau*  g*(tau*+{SENSOR_DELAY})")
        print("-" * len(header) + "-+------------------")
        for gain in GAINS:
            cells = [settle_fraction(gain, dt, jitter, seeds) for dt in DEAD_TIMES]
            tau_star = 0
            for dt, frac in zip(DEAD_TIMES, cells):
                if frac >= SETTLE_THRESHOLD:
                    tau_star = dt
                else:
                    break
            prod = gain * (tau_star + SENSOR_DELAY)
            cells_s = "".join(f"{c:>6.2f}" for c in cells)
            prod_s = f"{prod:>6.2f}" if tau_star else "   <1  (unstable at tau=1)"
            print(f"{gain:>6} | {cells_s} | {tau_star:>3}  {prod_s}")


if __name__ == "__main__":
    main()
