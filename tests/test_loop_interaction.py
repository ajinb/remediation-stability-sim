"""H3: individually-stable loops on a shared actuator destabilize; the
loop-interaction matrix (actuator mutex) restores coupled stability."""

from remediation_stability_sim import (
    Constructs,
    ControllerConfig,
    PlantConfig,
    run_multi_scenario,
    run_scenario,
)

# Disturbance chosen so damping caps do not bind: interference must come from
# the loops, not from magnitude clipping coincidences.
SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=25.0, noise_sd=0.5, sensor_delay=2)


def agent_controller():
    return ControllerConfig(gain=0.8, dead_time_mean=6, dead_time_jitter=0.2,
                            magnitude_jitter=0.2, act_threshold=5.0)


def legacy_autoscaler():
    return ControllerConfig(gain=0.6, dead_time_mean=2, dead_time_jitter=0.0,
                            magnitude_jitter=0.0, act_threshold=5.0, period=3)


def stable_constructs(interaction_matrix: bool) -> Constructs:
    return Constructs(hysteresis=True, engage_band=8.0, release_band=3.0,
                      damping=True, max_step=20.0,
                      cooldown=True, settle_steps=10,
                      deadtime_aware=True, interaction_matrix=interaction_matrix)


def test_each_loop_is_individually_stable():
    for ctrl in (agent_controller(), legacy_autoscaler()):
        r = run_scenario(plant=PlantConfig(**SPIKE), controller=ctrl,
                         constructs=stable_constructs(interaction_matrix=False),
                         steps=400, seed=0)
        assert r.reversals <= 1
        assert r.settling_time is not None


def test_coupled_loops_destabilize_without_coordination():
    harmed = 0
    for seed in range(20):
        r = run_multi_scenario(plant=PlantConfig(**SPIKE),
                               controllers=[agent_controller(), legacy_autoscaler()],
                               constructs=stable_constructs(interaction_matrix=False),
                               steps=400, seed=seed)
        harmed += r.reversals >= 2 or r.overshoot > 8.0 or r.settling_time is None
    assert harmed >= 14, f"coupled loops should interfere in most seeds ({harmed}/20)"


def test_interaction_matrix_restores_coupled_stability():
    stable = 0
    for seed in range(20):
        r = run_multi_scenario(plant=PlantConfig(**SPIKE),
                               controllers=[agent_controller(), legacy_autoscaler()],
                               constructs=stable_constructs(interaction_matrix=True),
                               steps=400, seed=seed)
        stable += r.reversals <= 1 and r.settling_time is not None and r.overshoot <= 8.0
    assert stable >= 18, f"mutex should stabilize almost all seeds ({stable}/20)"
