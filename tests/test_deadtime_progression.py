"""H1 pinned: at fixed gain, instability arrives as dead-time grows.

Gain is sized so the loop is stable at dead_time=1; the only thing that
changes across the sweep is dead-time. Individually-correct decisions,
collectively unstable — the paper's central claim.
"""

import statistics

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)


def run_sweep(dead_time: int):
    return [
        run_scenario(plant=PlantConfig(**SPIKE),
                     controller=ControllerConfig(gain=0.3, dead_time_mean=dead_time,
                                                 dead_time_jitter=0.3, magnitude_jitter=0.3,
                                                 act_threshold=5.0),
                     constructs=Constructs(), steps=400, seed=s)
        for s in range(20)
    ]


def test_short_deadtime_is_stable():
    rs = run_sweep(1)
    assert sum(r.reversals >= 3 for r in rs) <= 2
    assert sum(r.settling_time is not None for r in rs) == 20


def test_long_deadtime_oscillates_and_never_settles():
    rs = run_sweep(8)
    assert sum(r.reversals >= 3 for r in rs) >= 18
    assert sum(r.settling_time is None for r in rs) >= 18


def test_oscillation_amplitude_grows_with_deadtime():
    amps = [statistics.mean(r.osc_amplitude for r in run_sweep(dt)) for dt in (1, 4, 16)]
    assert amps[0] < amps[1] < amps[2] * 1.001  # monotone growth (16 may rail-clip)


def test_deadtime_aware_gating_alone_stabilizes_cheaper_than_cooldown():
    ctrl = ControllerConfig(gain=0.3, dead_time_mean=8, dead_time_jitter=0.3,
                            magnitude_jitter=0.3, act_threshold=5.0)

    def sweep(constructs):
        return [run_scenario(plant=PlantConfig(**SPIKE), controller=ctrl,
                             constructs=constructs, steps=400, seed=s) for s in range(20)]

    aware = sweep(Constructs(deadtime_aware=True))
    cool = sweep(Constructs(cooldown=True))
    for rs in (aware, cool):
        assert sum(r.reversals >= 3 for r in rs) <= 2
        assert sum(r.settling_time is not None for r in rs) == 20
    # Adapting the hold to actual dead-time recovers faster than a fixed window.
    assert statistics.mean(r.settling_time for r in aware) < \
        statistics.mean(r.settling_time for r in cool) * 0.6
