"""Pin the closed-form results of the stability analysis against numerics."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import theory as th  # noqa: E402


@pytest.mark.parametrize("L", range(0, 25))
def test_closed_form_boundary_matches_spectral_radius(L):
    g = th.g_star(L)
    assert th.spectral_radius(g * 0.995, L) < 1.0
    assert th.spectral_radius(g * 1.005, L) > 1.0


@pytest.mark.parametrize("L", range(1, 20))
def test_fastest_gain_attains_floor_and_floor_is_global(L):
    gf = th.g_fastest(L)
    assert th.spectral_radius(gf, L) == pytest.approx(L / (L + 1), abs=2e-4)
    grid = np.linspace(1e-4, th.g_star(L) * 1.5, 4001)
    radii = np.array([th.spectral_radius(g, L) for g in grid])
    assert radii.min() >= L / (L + 1) - 2e-4          # Gauss-Lucas lower bound
    assert grid[radii.argmin()] == pytest.approx(gf, rel=0.02)


def test_product_rule_is_sufficient_and_asymptotically_tight():
    products = [th.g_star(L) * (L + 1) for L in range(0, 400)]
    assert all(p > math.pi / 2 for p in products)
    assert all(a > b for a, b in zip(products, products[1:]))      # decreasing
    assert products[-1] - math.pi / 2 < 0.01


@pytest.mark.parametrize("K", range(0, 30))
def test_varying_delay_bound_is_inside_constant_delay_region(K):
    assert th.varying_delay_gain(K) <= th.g_star(K) + 1e-12


def test_predictor_full_gain_stable_iff_kappa_in_open_interval():
    for L in (1, 3, 6):
        for kappa in (0.05, 0.5, 1.0, 1.5, 1.95):
            assert th.predictor_radius(1.0, L, kappa) < 1.0
        for kappa in (2.05, 2.5):
            assert th.predictor_radius(1.0, L, kappa) > 1.0


def test_repair_floor_grows_linearly_with_delay():
    for L in (1, 2, 4, 8, 16):
        steps = th.repair_floor_steps(L, 20.0)
        assert L * math.log(20) < steps < (L + 0.5) * math.log(20)
