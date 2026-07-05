"""Reproduce the simulation study for "Stable by Design" (§6).

Three experiments, 20 seeds each, 400-step scenarios, persistent step
disturbance observed through a 2-step-delayed sensor:

  E1 (H1)  greedy per-decision-correct controller vs. growing dead-time
  E2 (H2)  incremental stable-by-design constructs at LLM-like dead-time (8)
  E3 (H3)  two individually-stable loops on a shared actuator, +/- mutex

Usage: python examples/paper_study.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import statistics

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_multi_scenario, run_scenario

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)
SPIKE_COUPLED = dict(SPIKE, disturbance=25.0)


def controller(dead_time: int) -> ControllerConfig:
    # Gain sized so the loop is stable at short dead-time: the instability in
    # E1 then comes from dead-time growth alone, not from an over-tuned gain.
    return ControllerConfig(gain=0.3, dead_time_mean=dead_time, dead_time_jitter=0.3,
                            magnitude_jitter=0.3, act_threshold=5.0)


def agg(results):
    n = len(results)
    osc = sum(r.reversals >= 3 for r in results)
    settled = [r.settling_time for r in results if r.settling_time is not None]
    return {
        "oscillating": f"{osc}/{n}",
        "reversals": statistics.mean(r.reversals for r in results),
        "amplitude": statistics.mean(r.osc_amplitude for r in results),
        "overshoot": statistics.mean(r.overshoot for r in results),
        "settle_rate": f"{len(settled)}/{n}",
        "mttr": statistics.mean(settled) if settled else float("nan"),
    }


def row(label, results):
    a = agg(results)
    print(f"{label:<28} osc={a['oscillating']:>5}  rev={a['reversals']:>5.1f}  "
          f"amp={a['amplitude']:>6.1f}  over={a['overshoot']:>5.1f}  "
          f"settled={a['settle_rate']:>5}  MTTR={a['mttr']:>6.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    seeds = range(args.seeds)

    print("=== E1 (H1): greedy baseline vs dead-time ===")
    for dt in (1, 2, 4, 8, 16):
        rs = [run_scenario(plant=PlantConfig(**SPIKE), controller=controller(dt),
                           constructs=Constructs(), steps=400, seed=s) for s in seeds]
        row(f"dead_time={dt}", rs)

    print("\n=== E2 (H2): constructs at dead_time=8, isolated and combined ===")
    variants = [
        ("baseline (greedy)", Constructs()),
        ("hysteresis+damping only", Constructs(hysteresis=True, damping=True)),
        ("cooldown alone", Constructs(cooldown=True)),
        ("deadtime-aware alone", Constructs(deadtime_aware=True)),
        ("all four constructs", Constructs(hysteresis=True, damping=True,
                                           cooldown=True, deadtime_aware=True)),
    ]
    for label, cons in variants:
        rs = [run_scenario(plant=PlantConfig(**SPIKE), controller=controller(8),
                           constructs=cons, steps=400, seed=s) for s in seeds]
        row(label, rs)

    print("\n=== E3 (H3): coupled loops on a shared actuator (disturbance=25) ===")
    agent = ControllerConfig(gain=0.8, dead_time_mean=6, dead_time_jitter=0.2,
                             magnitude_jitter=0.2, act_threshold=5.0)
    legacy = ControllerConfig(gain=0.6, dead_time_mean=2, dead_time_jitter=0.0,
                              magnitude_jitter=0.0, act_threshold=5.0, period=3)
    stable = dict(hysteresis=True, engage_band=8.0, release_band=3.0, damping=True,
                  max_step=20.0, cooldown=True, settle_steps=10, deadtime_aware=True)
    for label, ctrls, matrix in [
        ("agent alone", [agent], False),
        ("autoscaler alone", [legacy], False),
        ("coupled, no coordination", [agent, legacy], False),
        ("coupled + interaction mtx", [agent, legacy], True),
    ]:
        rs = [run_multi_scenario(plant=PlantConfig(**SPIKE_COUPLED), controllers=ctrls,
                                 constructs=Constructs(**stable, interaction_matrix=matrix),
                                 steps=400, seed=s) for s in seeds]
        row(label, rs)


if __name__ == "__main__":
    main()
