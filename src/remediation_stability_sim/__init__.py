"""remediation-stability-sim: discrete-time Remediation Control Loop simulator.

Companion artifact for "Stable by Design: A Control-Theoretic Account of
AI-Driven Self-Healing Remediation Loops". Models the paper's RCL (§4): an
integrator plant observed through a delayed sensor, corrected by a
probabilistic controller with dead-time, optionally wrapped in the §5
stable-by-design constructs.
"""

from .sim import Constructs, ControllerConfig, PlantConfig, Result, run_multi_scenario, run_scenario

__all__ = [
    "Constructs",
    "ControllerConfig",
    "PlantConfig",
    "Result",
    "run_scenario",
    "run_multi_scenario",
]
