# Forbidden Sets: Theoretical RL Research Library

A theory-first Python library for experimentally isolating the structural causes of exponential complexity in deterministic RL under partial observability.

## Research Context

This library implements the theoretical framework from research on **elimination-based learning** in deterministic MDPs. The key insight is that learning can proceed by accumulating *forbidden* (observation, action) pairs rather than estimating values, and the memory complexity of this process depends critically on the sufficiency of the state representation.

### Core Findings

1. **Polynomial Scaling Under Sufficient Representation**: When observations uniquely identify states, the forbidden set |F| grows polynomially with the environment diameter D.

2. **Exponential Blowup Under Aliasing**: When multiple states map to the same observation (perceptual aliasing), a stateless agent may need exponentially many forbidden pairs.

3. **Recovery via History**: Augmenting observations with a single step of history often collapses exponential aliasing back to polynomial growth.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "The Complexity of Learned Forbidden Sets"

# Set PYTHONPATH (or install as package)
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Requirements**: Python 3.10+ (uses frozen dataclasses, TYPE_CHECKING, etc.)

Optional: `matplotlib` for plotting utilities.

## Quick Start

```python
from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.observation import ObservationMapping
from forbidden_sets.agents.forbidden import ForbiddenSetAgent
from forbidden_sets.harness.rollout import DeterministicRollout

# Create environment
env = CorridorMDP(diameter=20, num_actions=4)
obs_mapping = ObservationMapping.identity(env.states)
agent = ForbiddenSetAgent(env.actions, history_depth=0)
rollout = DeterministicRollout(max_steps=100)

# Run learning
for episode in range(100):
    result = rollout.run_episode(env, agent, obs_mapping, episode)
    if result.succeeded:
        print(f"Episode {episode}: SUCCESS, |F| = {agent.memory_size}")
```

## Running Experiments

The library includes pre-configured experiments from the papers:

```python
from forbidden_sets.experiments.polynomial_growth import run_polynomial_scaling_experiment
from forbidden_sets.experiments.aliasing_stress import run_aliasing_stress_experiment
from forbidden_sets.experiments.history_recovery import run_history_recovery_experiment

# Experiment A: Polynomial growth without aliasing
result_a = run_polynomial_scaling_experiment(diameters=[10, 20, 40, 80])

# Experiment B: Aliasing stress test
result_b = run_aliasing_stress_experiment(alias_factors=[1, 2, 4, 8, 16])

# Experiment C: Recovery via history
result_c = run_history_recovery_experiment(history_depth=1)
```

## Package Structure

```
forbidden_sets/
├── core/              # Foundational types and constraints
│   ├── types.py       # State, Observation, Action, Trajectory, etc.
│   ├── errors.py      # TheoreticalViolationError and subclasses
│   └── invariants.py  # Enforcement functions
├── environments/      # Deterministic MDP implementations
│   ├── base.py        # DeterministicMDP abstract class
│   ├── observation.py # ObservationMapping with aliasing control
│   ├── corridor.py    # Linear chain environment
│   └── conflicting_graph.py  # Adversarial aliasing environment
├── agents/            # Learning agents
│   ├── base.py        # Agent protocol
│   ├── stateless.py   # StatelessAgent
│   ├── history.py     # FiniteHistoryAgent
│   └── forbidden.py   # ForbiddenSetAgent (core)
├── metrics/           # Measurement tools
│   ├── constraint_size.py    # |F| growth tracking
│   ├── false_elimination.py  # Error detection
│   ├── feasibility.py        # Success/failure rates
│   └── seed_invariance.py    # Determinism verification
├── harness/           # Experiment infrastructure
│   ├── rollout.py     # DeterministicRollout engine
│   ├── experiment.py  # ExperimentRunner, ExperimentConfig
│   └── plotting.py    # Log-log visualization
└── experiments/       # Paper reproductions
    ├── polynomial_growth.py  # Experiment A
    ├── aliasing_stress.py    # Experiment B
    ├── history_recovery.py   # Experiment C
    └── representation.py     # Experiments D, E, F, H
```

## Key Concepts

### Forbidden Sets

The core data structure is a set of (observation, action) pairs that have been determined to lead to failure:

```python
F ⊆ O × A
```

Learning proceeds by accumulating pairs into F. The agent never estimiates values—it simply eliminates bad actions.

### Deterministic Dynamics

All environments enforce deterministic transitions:

```python
s' = T(s, a)  # No stochasticity
```

This enables precise analysis of memory requirements without confounding from noise.

### Aliasing

Aliasing occurs when multiple states map to the same observation:

```python
φ(s₁) = φ(s₂) = o  but  π*(s₁) ≠ π*(s₂)
```

Under aliasing, a stateless agent cannot distinguish states that require different optimal actions, leading to memory blowup.

### History Augmentation

Using history tuples as keys resolves many aliasing cases:

```python
key = (o_{t-1}, o_t)  # Single-step history
```

This often restores polynomial memory growth without full belief state tracking.

## Design Philosophy

This library deliberately **does not** include:

- Value function estimation
- Stochastic policies
- Reward curves or regret metrics
- Hidden memory structures

These would be inappropriate for the theoretical framework being studied. All agent memory is explicit and externally accessible.

## API Reference

### Core Types

- `State`, `Observation`, `Action`: Basic integer newtypes
- `Trajectory`: Complete episode recording
- `ForbiddenPair`: Detailed record of a forbidden pair

### Environments

- `DeterministicMDP`: Abstract base with determinism verification
- `CorridorMDP`: Linear chain with configurable diameter
- `ConflictingGraphMDP`: Adversarial aliasing testbed

### Agents

- `StatelessAgent`: Reactive elimination agent
- `FiniteHistoryAgent`: Bounded history for disambiguation
- `ForbiddenSetAgent`: Core elimination learner with monotonicity

### Metrics

- `ConstraintSizeTracker`: Track |F| growth with polynomial fitting
- `FalseEliminationTracker`: Detect incorrect eliminations
- `FeasibilityTracker`: Success/failure/infeasibility rates

### Harness

- `DeterministicRollout`: Execute episodes with full control
- `ExperimentRunner`: Run complete experiments with metrics
- `ExperimentConfig`/`ExperimentResult`: Configuration and results

## License

[Your License Here]

## Citation

If you use this library in your research, please cite:

```bibtex
@software{forbidden_sets,
  title={Forbidden Sets: Theoretical RL Research Library},
  author={Your Name},
  year={2026},
  description={Python library for studying elimination-based learning complexity}
}
```
