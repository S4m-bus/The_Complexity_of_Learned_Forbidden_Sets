#!/usr/bin/env python3
"""
Example: Basic usage of the Forbidden Sets library.

This script demonstrates:
1. Creating an environment (CorridorMDP)
2. Setting up observation mapping with aliasing
3. Creating a ForbiddenSetAgent
4. Running episodes and observing learning
5. Analyzing the forbidden set growth

Run from the project root:
    python -m examples.basic_usage
or:
    set PYTHONPATH=%cd% && python examples/basic_usage.py
"""

from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.observation import ObservationMapping
from forbidden_sets.agents.forbidden import ForbiddenSetAgent
from forbidden_sets.harness.rollout import DeterministicRollout
from forbidden_sets.metrics.constraint_size import ConstraintSizeTracker


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    print("=" * 60)
    print("Forbidden Sets Library - Basic Usage Example")
    print("=" * 60)
    
    # Create a corridor environment
    # - diameter: length of the corridor (start → goal distance)
    # - num_actions: number of actions available at each state
    diameter = 20
    num_actions = 4
    
    env = CorridorMDP(diameter=diameter, num_actions=num_actions)
    print(f"\nEnvironment: CorridorMDP")
    print(f"  Diameter (D): {diameter}")
    print(f"  States: {env.num_states}")
    print(f"  Actions: {env.num_actions}")
    print(f"  Goal state: {list(env.goal_states)}")
    
    # Create observation mapping (identity = no aliasing)
    obs_mapping = ObservationMapping.identity(env.states)
    print(f"\nObservation Mapping: Identity (no aliasing)")
    print(f"  Observations: {len(obs_mapping.observations)}")
    
    # Create the learning agent
    agent = ForbiddenSetAgent(actions=env.actions, history_depth=0)
    print(f"\nAgent: ForbiddenSetAgent")
    print(f"  History depth: {agent.history_depth}")
    print(f"  Initial |F|: {agent.memory_size}")
    
    # Create rollout engine
    rollout = DeterministicRollout(max_steps=100)
    
    # Create tracker
    tracker = ConstraintSizeTracker(environment_diameter=diameter)
    
    # =========================================================================
    # Learning Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running 50 episodes...")
    print("-" * 60)
    
    num_episodes = 50
    successes = 0
    
    for episode in range(num_episodes):
        result = rollout.run_episode(env, agent, obs_mapping, episode)
        
        if result.succeeded:
            successes += 1
        
        # Track forbidden set size
        tracker.log(episode, agent.memory_size)
        
        # Print progress every 10 episodes
        if episode % 10 == 0 or episode == num_episodes - 1:
            print(f"  Episode {episode:3d}: {result.outcome.name:10s} | "
                  f"|F| = {agent.memory_size:3d} | "
                  f"Success rate: {successes/(episode+1):.1%}")
    
    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("Analysis")
    print("-" * 60)
    
    print(f"\nFinal Results:")
    print(f"  Final |F|: {agent.memory_size}")
    print(f"  Total successes: {successes}/{num_episodes} ({successes/num_episodes:.1%})")
    
    # Fit polynomial growth model
    fit = tracker.fit_polynomial()
    print(f"\nGrowth Analysis:")
    print(f"  {fit}")
    
    if fit.is_polynomial:
        print(f"  ✓ Polynomial growth confirmed (exponent ≤ {fit.polynomial_threshold})")
    else:
        print(f"  ✗ Growth exceeds polynomial bounds")
    
    # Show forbidden set structure
    print(f"\nForbidden Set Structure:")
    memory = agent.memory
    print(f"  Total pairs: {memory.size}")
    print(f"  Distinct keys: {len(memory.by_key)}")
    
    # Show a few examples
    print(f"\n  Sample forbidden pairs:")
    for i, record in enumerate(memory.detailed_records[:5]):
        print(f"    ({record.observation}, action={record.action}) "
              f"- forbidden at episode {record.episode_forbidden}")
    if memory.size > 5:
        print(f"    ... and {memory.size - 5} more")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
