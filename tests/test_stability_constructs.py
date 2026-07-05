"""H1/H2: dead-time destabilizes a greedy controller; §5 constructs restore stability."""

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)


def slow_probabilistic_controller():
    """An LLM-like controller: large variable dead-time, jittered magnitude."""
    return ControllerConfig(gain=0.8, dead_time_mean=8, dead_time_jitter=0.3,
                            magnitude_jitter=0.3, act_threshold=5.0)


def all_constructs():
    return Constructs(hysteresis=True, engage_band=8.0, release_band=3.0,
                      damping=True, max_step=20.0,
                      cooldown=True, settle_steps=10,
                      deadtime_aware=True)


def test_baseline_oscillates_under_large_deadtime():
    r = run_scenario(plant=PlantConfig(**SPIKE), controller=slow_probabilistic_controller(),
                     constructs=Constructs(), steps=400, seed=0)
    assert r.reversals >= 3, f"expected sustained oscillation, got {r.reversals} reversals"
    assert r.osc_amplitude > 60.0
    assert r.settling_time is None or r.settling_time > 200


def test_constructs_stabilize_large_deadtime():
    r = run_scenario(plant=PlantConfig(**SPIKE), controller=slow_probabilistic_controller(),
                     constructs=all_constructs(), steps=400, seed=0)
    assert r.reversals <= 1, f"expected stability, got {r.reversals} reversals"
    assert r.settling_time is not None, "stabilized loop must settle"
    assert r.overshoot <= 10.0


def test_stability_holds_across_seeds():
    unstable, stable = 0, 0
    for seed in range(20):
        base = run_scenario(plant=PlantConfig(**SPIKE), controller=slow_probabilistic_controller(),
                            constructs=Constructs(), steps=400, seed=seed)
        cons = run_scenario(plant=PlantConfig(**SPIKE), controller=slow_probabilistic_controller(),
                            constructs=all_constructs(), steps=400, seed=seed)
        unstable += base.reversals >= 3
        stable += cons.reversals <= 1 and cons.settling_time is not None
    assert unstable >= 16, f"baseline should oscillate in most seeds ({unstable}/20)"
    assert stable >= 18, f"constructs should stabilize almost all seeds ({stable}/20)"
