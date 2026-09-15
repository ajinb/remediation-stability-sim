"""Closed-form and numerical stability results for delayed remediation loops.

Every-step proportional remediation with loop delay L (in control periods):

    e[k+1] = e[k] - g * e[k-L]                              (delay-integrator loop)

and its fractional-actuation variant, where a share phi of each period passes
before new capacity is effective:

    e[k+1] = e[k] - g(1-phi) e[k-L] - g phi e[k-L-1]

Results implemented here (the paper's Section 4):

* g_star(L)          exact stability boundary 2 sin(pi / (2(2L+1)))  [Levin-May]
* g_fastest(L)       gain minimising the spectral radius, L^L / (L+1)^(L+1)
* rho_floor(L)       that minimum, L / (L+1)                          [Prop. 2]
* product_rule       g (L+1) < pi/2 is sufficient for every L          [Cor. 3]
* varying_delay_ok   g (K+1) < 3/2 + 1/(2(K+1)) for delays <= K       [Erbe-Xia-Yu, Yu]
* predictor/gated    characteristic polynomials under plant-gain mismatch kappa
"""

from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- basics
def char_poly(g: float, L: int, phi: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """Coefficients (highest degree first) of z^{L+2} - z^{L+1} + a z + b."""
    n = L + 2
    c = np.zeros(n + 1)
    c[0] = 1.0
    c[1] -= 1.0
    c[n - 1] += g * kappa * (1.0 - phi)
    c[n] += g * kappa * phi
    return c


def spectral_radius(g: float, L: int, phi: float = 0.0, kappa: float = 1.0) -> float:
    return float(np.max(np.abs(np.roots(char_poly(g, L, phi, kappa)))))


def g_star(L: int) -> float:
    """Exact stability boundary of e[k+1] = e[k] - g e[k-L] (0 < g < g_star)."""
    return 2.0 * math.sin(math.pi / (2.0 * (2 * L + 1)))


def g_star_numeric(L: int, phi: float = 0.0, hi: float = 2.0) -> float:
    """Boundary by bisection on the spectral radius (phi in [0, 1), L >= 1)."""
    lo = 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, L, phi) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def g_fastest(L: int) -> float:
    """Gain at which the spectral radius is smallest (a double root at L/(L+1))."""
    L = int(L)
    if L == 0:
        return 1.0
    return (L / (L + 1)) ** L / (L + 1)   # float form: L**L overflows int64 past L=15


def rho_floor(L: int) -> float:
    """Smallest achievable spectral radius for every-step proportional control."""
    return L / (L + 1)


def repair_floor_steps(L: int, reduction: float) -> float:
    """Fewest control periods in which the error envelope can shrink by `reduction`x."""
    if L == 0:
        return 1.0
    return math.log(reduction) / math.log((L + 1) / L)


def product_rule_ok(g: float, L: int) -> bool:
    """Sufficient condition g (L+1) < pi/2 (Corollary 3)."""
    return g * (L + 1) < math.pi / 2


def varying_delay_ok(g: float, K: int) -> bool:
    """Sufficient condition for delays varying in [0, K] (3/2-type criterion)."""
    return g * (K + 1) < 1.5 + 1.0 / (2 * (K + 1))


def varying_delay_gain(K: int) -> float:
    return (1.5 + 1.0 / (2 * (K + 1))) / (K + 1)


# ---------------------------------------------------------------------- constructs
def predictor_poly(g: float, L: int, kappa: float) -> np.ndarray:
    """Smith-predictor loop with exact delay model and plant-gain mismatch kappa:
    z^{L+1} - (1-g) z^L - g (1 - kappa) = 0."""
    c = np.zeros(L + 2)
    c[0] = 1.0
    c[1] = -(1.0 - g)
    c[-1] += -g * (1.0 - kappa)
    return c


def predictor_radius(g: float, L: int, kappa: float) -> float:
    return float(np.max(np.abs(np.roots(predictor_poly(g, L, kappa)))))


def gated_cycle_factor(g: float, kappa: float) -> float:
    """Error multiplier per decision cycle for event-gated pulse remediation."""
    return abs(1.0 - g * kappa)


def cooldown_cycle_factor(g: float, kappa: float, hold: int) -> float:
    """Integrating plant, absolute command held for `hold` periods before re-deciding
    on fresh data: error multiplier per cycle."""
    return abs(1.0 - hold * g * kappa)


def fractional_phi_boundary_table(Ls=(1, 2, 3, 4, 5, 6, 8), phis=(0.0, 0.2, 0.3, 0.4)) -> list[dict]:
    rows = []
    for L in Ls:
        row = {"L": L, "g_star_exact": g_star(L)}
        for p in phis:
            row[f"g_star_phi{p}"] = g_star_numeric(L, p)
        rows.append(row)
    return rows
