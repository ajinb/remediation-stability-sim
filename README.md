# remediation-stability-sim

> Stability of AI-driven remediation loops: closed-form limits, simulation studies, a Kubernetes
> testbed with a digital twin, and LLM controllers in the loop.

This repository is the companion artifact for two documents:

1. **Delay-bounded stability of AI-driven remediation loops: analysis and Kubernetes experiments with
   large language model controllers** (journal manuscript, 2026). Everything under `analysis/`,
   `cluster/`, `llm/`, `paper/` and `data/` belongs to it.
2. **Stable by Design: A Control-Theoretic Account of AI-Driven Self-Healing Remediation Loops**
   (preprint v0.3, 2026). The simulator in `src/` and the scripts in `examples/` belong to it; the journal
   manuscript reuses the simulator for its study E0a.

## The one-line result

A controller that corrects a fraction `g` of the deficit it sees, with a loop delay of `L` periods,
follows `e[k+1] = e[k] - g*e[k-L]`. It is stable if and only if `g < 2 sin(pi / (2(2L+1)))`; no gain
decays faster than `L/(L+1)` per period; and `g(L+1) < pi/2` is sufficient at every delay. In time units:
**plan to drain what you see over more than (2/pi) x (loop delay + period).**

## Layout

| Path | What it is |
|---|---|
| `analysis/theory.py` | closed forms and numerics: boundary, fastest gain, decay floor, product rule, varying-delay bound, predictor and construct polynomials |
| `analysis/e0_studies.py` | E0a (prior frontier vs theory), E0b (random latency, overlapping decisions), E0c (constructs under uncertainty), E0d (sequential agent loops) |
| `cluster/plant.yaml.tpl` | queue-worker plant: Redis list + worker Deployment whose replica count is the actuator |
| `cluster/lab.py` | host harness: exact-rate producer, delayed sensor, injected dead time, policies, JSONL traces |
| `cluster/twin.py` | virtual-time digital twin with the same interface as the real plant |
| `cluster/calibrate.py`, `cluster/identify.py` | calibration, and least-squares identification of plant gain and actuation lag from traces |
| `cluster/experiments.py`, `cluster/analyze.py` | E1 (frontier) and E2 (constructs) grids; per-run metrics |
| `llm/claude_cli.py`, `llm/probe.py`, `llm/closed_loop.py` | E3a open-loop probe and E3b closed-loop runs with Claude models via the Claude Code CLI |
| `paper/make_figures.py`, `paper/tables.py`, `paper/build_bib.py` | every figure, table and the reference list, regenerated from `data/` and live records (`paper/refs/`) |
| `data/` | all traces and study outputs, including discarded runs and why they were discarded |
| `tests/` | 93 tests, including the closed-form results pinned against numerics |

## Reproduce

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest                                               # theory and the preprint's simulator
python analysis/e0_studies.py --out data/e0          # E0a-E0d (E0b/E0c take several minutes)
python paper/tables.py --data data --out paper/generated
python paper/make_figures.py --data data --out paper/figures
```

### Kubernetes testbed (E1, E2)

Requires Docker and [kind](https://kind.sigs.k8s.io/) 0.33 or later. The image `redis:7-alpine` is
loaded into the cluster so no pulls happen during runs.

```bash
kind create cluster --config cluster/kind-config.yaml --kubeconfig ~/.local/share/stability-lab/kubeconfig
kind load docker-image redis:7-alpine --name stability-lab
export KUBECONFIG=~/.local/share/stability-lab/kubeconfig
python cluster/calibrate.py lab-1
python cluster/experiments.py e1 --slot 1 --shard 0/2 --reps 2 &
python cluster/experiments.py e1 --slot 2 --shard 1/2 --reps 2
python cluster/experiments.py e2 --slot 1 --shard 0/2 --reps 2 &
python cluster/experiments.py e2 --slot 2 --shard 1/2 --reps 2
```

Keep the host otherwise idle: four runs in the original study stalled while CPU-heavy simulations ran on
the same machine and were discarded (`data/discarded/e1/README.txt`). Runs whose control periods start more
than 1 s late are not valid.

### LLM controllers (E3a, E3b)

Uses the Claude Code CLI in headless mode (`claude -p`) with tools, plugins and MCP servers disabled and a
replaced system prompt. Each call counts against the account it runs under.

```bash
python llm/probe.py --models claude-haiku-4-5,claude-sonnet-5 --reps 2
python llm/closed_loop.py --model claude-sonnet-5 --condition stateless --slot 1 --rep 0
python llm/closed_loop.py --model claude-sonnet-5 --condition pulse --slot 1 --rep 0
```

## The preprint's simulator

```bash
python examples/paper_study.py      # preprint E1-E3
python examples/frontier_study.py   # preprint E4, the frontier that analysis/e0_studies.py E0a explains
```

## License

Apache-2.0 (see `LICENSE`), for code and data.
