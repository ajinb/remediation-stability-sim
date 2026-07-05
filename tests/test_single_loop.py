"""Single-loop behavior of the Remediation Control Loop simulator."""

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario


def quiet_plant(**overrides):
    cfg = dict(setpoint=50.0, disturbance_at=None, disturbance=0.0,
               noise_sd=0.5, sensor_delay=0)
    cfg.update(overrides)
    return PlantConfig(**cfg)


def spiked_plant(**overrides):
    cfg = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0,
               noise_sd=0.5, sensor_delay=2)
    cfg.update(overrides)
    return PlantConfig(**cfg)


def fast_controller(**overrides):
    cfg = dict(gain=0.8, dead_time_mean=1, dead_time_jitter=0.0,
               magnitude_jitter=0.0, act_threshold=5.0)
    cfg.update(overrides)
    return ControllerConfig(**cfg)


def test_same_seed_same_result():
    a = run_scenario(plant=spiked_plant(), controller=fast_controller(),
                     constructs=Constructs(), steps=400, seed=7)
    b = run_scenario(plant=spiked_plant(), controller=fast_controller(),
                     constructs=Constructs(), steps=400, seed=7)
    assert a.trajectory == b.trajectory
    assert a.actions == b.actions


def test_no_disturbance_means_no_actions():
    r = run_scenario(plant=quiet_plant(), controller=fast_controller(),
                     constructs=Constructs(), steps=400, seed=0)
    assert r.actions == 0
    assert r.reversals == 0


def test_zero_deadtime_baseline_settles_without_oscillation():
    r = run_scenario(plant=spiked_plant(sensor_delay=0), controller=fast_controller(),
                     constructs=Constructs(), steps=400, seed=0)
    assert r.actions >= 1
    assert r.reversals <= 1
    assert r.settling_time is not None
    assert r.settling_time < 30
