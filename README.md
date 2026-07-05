# remediation-stability-sim

> Discrete-time simulator for **AI-driven self-healing remediation loops**: reproduce the instability modes (oscillation, overshoot, runaway, loop interaction) and the stable-by-design constructs that tame them.

Companion artifact for the paper **"Stable by Design: A Control-Theoretic Account of AI-Driven Self-Healing Remediation Loops."**

## Model

- **Plant** — a bounded scalar state (utilization-like) holding a setpoint until a persistent step disturbance shifts it; remediations are persistent shifts back.
- **Sensor** — reports the plant `sensor_delay` steps late, with noise.
- **Controller** — proportional remediation policy with *probabilistic* magnitude (lognormal jitter) and *probabilistic* dead-time (reasoning latency): the paper's LLM-in-the-loop controller. Direction is always correct given the measurement — decisions are individually correct by construction.
- **Constructs (§5 of the paper)** — hysteresis gates, action damping, settling-time cooldowns, dead-time-aware gating, and the loop-interaction matrix (actuator mutex held through the correction's settling window).

## Reproduce the paper's study

```bash
pip install -e ".[dev]"
pytest                          # 13 tests: H1, H2, H3 each pinned
python examples/paper_study.py  # E1/E2/E3 tables (20 seeds per cell)
```

Headline results (400-step scenarios, 20 seeds per cell):

- **E1 (H1)** — at fixed gain, growing dead-time alone flips the loop from stable (τ=1: 1/20 oscillating, MTTR ≈ 15) to rail-to-rail oscillation that never settles (τ≥4: 20/20 oscillating).
- **E2 (H2)** — hysteresis+damping alone do *not* stabilize a dead-time-driven oscillation; the temporal constructs are each individually sufficient, and **dead-time-aware gating settles ~3× faster than a fixed cooldown** (MTTR ≈ 40 vs ≈ 117) because it adapts its hold to the controller's actual dead-time.
- **E3 (H3)** — two individually-stable loops sharing an actuator destabilize each other (overshoot 8.9 vs 1.3; 13/20 settle); an actuator mutex held through the settling window restores single-loop behavior (20/20 settle).

## Not in scope

The stochastic controller abstraction approximates an LLM's variability; it is not an LLM. The simulator establishes the *mechanisms*; production validation is future work (see paper §7).

## License

MIT
