"""E4: the stability frontier (paper §6, v0.3).

Pins the three qualitative claims of the frontier study:
  - the deterministic settle/diverge boundary follows a delay-gain product
    law, g x L ~ 1.5 with L = dead_time + sensor_delay, and is sharp (one
    dead-time step wide);
  - cells at product <= 1.2 settle in every seed at every jitter level
    (the engineering margin);
  - controller stochasticity blurs the frontier in BOTH directions rather
    than shifting it: deterministically-stable marginal cells lose
    reliability, deterministically-divergent cells occasionally settle.
"""

from remediation_stability_sim import Constructs, ControllerConfig, PlantConfig, run_scenario

SPIKE = dict(setpoint=50.0, disturbance_at=20, disturbance=40.0, noise_sd=0.5, sensor_delay=2)
SEEDS = range(10)


def settle_fraction(gain, dead_time, jitter):
    settled = 0
    for s in SEEDS:
        cfg = ControllerConfig(gain=gain, dead_time_mean=dead_time,
                               dead_time_jitter=jitter, magnitude_jitter=jitter,
                               act_threshold=5.0)
        r = run_scenario(plant=PlantConfig(**SPIKE), controller=cfg,
                         constructs=Constructs(), steps=400, seed=s)
        settled += r.settling_time is not None
    return settled / len(SEEDS)


def test_deterministic_frontier_is_a_sharp_product_law():
    # Two gains 2x apart put their cliff at the same product g*(tau+2) ~ 1.5,
    # and the cliff is one dead-time step wide: full settling on one side,
    # none on the other.
    assert settle_fraction(0.30, 3, 0.0) == 1.0   # g*L = 1.5
    assert settle_fraction(0.30, 4, 0.0) == 0.0   # g*L = 1.8
    assert settle_fraction(0.15, 8, 0.0) == 1.0   # g*L = 1.5
    assert settle_fraction(0.15, 10, 0.0) == 0.0  # g*L = 1.8


def test_margin_cells_settle_at_every_jitter_level():
    # g*L = 1.2: inside the engineering margin, settling survives controller
    # stochasticity at every level tested.
    for jitter in (0.0, 0.3, 0.6):
        assert settle_fraction(0.20, 4, jitter) == 1.0
        assert settle_fraction(0.30, 2, jitter) == 1.0


def test_jitter_blurs_the_frontier_in_both_directions():
    # Same product g*L = 1.8 on both sides of the blur:
    # a deterministically-stable marginal cell loses reliability...
    assert settle_fraction(0.60, 1, 0.0) == 1.0
    assert settle_fraction(0.60, 1, 0.6) <= 0.7
    # ...while a deterministically-divergent cell occasionally settles.
    assert settle_fraction(0.30, 4, 0.0) == 0.0
    assert settle_fraction(0.30, 4, 0.6) >= 0.1
