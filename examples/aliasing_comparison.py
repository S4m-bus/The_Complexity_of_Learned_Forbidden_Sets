#!/usr/bin/env python3
"""
Example: Comparing aliasing effects on memory growth.

This script demonstrates:
1. How aliasing affects forbidden set size
2. How history augmentation recovers polynomial scaling
3. Side-by-side comparison of different agent configurations

Run from the project root:
    python -m examples.aliasing_comparison
"""

from forbidden_sets.environments.corridor import CorridorMDP
from forbidden_sets.environments.observation import ObservationMapping
from forbidden_sets.agents.forbidden import ForbiddenSetAgent
from forbidden_sets.harness.rollout import DeterministicRollout


def run_with_config(diameter: int, alias_factor: int, history_depth: int, 
                    num_episodes: int = 100) -> dict:
    """Run an experiment with given configuration."""
    # Create environment
    env = CorridorMDP(diameter=diameter, num_actions=4)
    
    # Create observation mapping with controlled aliasing
    if alias_factor == 1:
        obs_mapping = ObservationMapping.identity(env.states)
    else:
        obs_mapping = ObservationMapping.floor_division(
            num_states=env.num_states,
            alias_factor=alias_factor
        )
    
    # Create agent
    agent = ForbiddenSetAgent(actions=env.actions, history_depth=history_depth)
    
    # Run episodes
    rollout = DeterministicRollout(max_steps=100)
    successes = 0
    
    for episode in range(num_episodes):
        result = rollout.run_episode(env, agent, obs_mapping, episode)
        if result.succeeded:
            successes += 1
    
    return {
        'alias_factor': alias_factor,
        'history_depth': history_depth,
        'final_F': agent.memory_size,
        'success_rate': successes / num_episodes,
        'num_keys': agent.num_keys,
    }


def main():
    print("=" * 70)
    print("Aliasing Comparison: Effect of Observation Collapse on Memory Growth")
    print("=" * 70)
    
    diameter = 40
    num_episodes = 200
    
    print(f"\nEnvironment: CorridorMDP with D={diameter}")
    print(f"Episodes per configuration: {num_episodes}")
    
    # Configuration sets to compare
    configs = [
        # (alias_factor, history_depth, label)
        (1, 0, "No aliasing, stateless"),
        (4, 0, "4x aliasing, stateless"),
        (8, 0, "8x aliasing, stateless"),
        (4, 1, "4x aliasing, history=1"),
        (8, 1, "8x aliasing, history=1"),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Configuration':<35} | {'|F|':>8} | {'Keys':>8} | {'Success':>10}")
    print("-" * 70)
    
    results = []
    for alias_factor, history_depth, label in configs:
        result = run_with_config(
            diameter=diameter,
            alias_factor=alias_factor,
            history_depth=history_depth,
            num_episodes=num_episodes
        )
        result['label'] = label
        results.append(result)
        
        print(f"{label:<35} | {result['final_F']:>8} | "
              f"{result['num_keys']:>8} | {result['success_rate']:>10.1%}")
    
    print("-" * 70)
    
    # Analysis
    print("\nAnalysis:")
    
    # Compare stateless with and without aliasing
    no_alias = results[0]
    alias_4 = results[1]
    alias_8 = results[2]
    
    print(f"\n1. Effect of aliasing (stateless agent):")
    print(f"   No aliasing: |F| = {no_alias['final_F']}")
    print(f"   4x aliasing: |F| = {alias_4['final_F']} "
          f"({alias_4['final_F']/no_alias['final_F']:.1f}x)")
    print(f"   8x aliasing: |F| = {alias_8['final_F']} "
          f"({alias_8['final_F']/no_alias['final_F']:.1f}x)")
    
    # Compare history augmentation
    alias_4_hist = results[3]
    alias_8_hist = results[4]
    
    print(f"\n2. Effect of history augmentation:")
    print(f"   4x aliasing stateless: |F| = {alias_4['final_F']}, "
          f"success = {alias_4['success_rate']:.1%}")
    print(f"   4x aliasing history=1: |F| = {alias_4_hist['final_F']}, "
          f"success = {alias_4_hist['success_rate']:.1%}")
    print(f"   8x aliasing stateless: |F| = {alias_8['final_F']}, "
          f"success = {alias_8['success_rate']:.1%}")
    print(f"   8x aliasing history=1: |F| = {alias_8_hist['final_F']}, "
          f"success = {alias_8_hist['success_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("Key Insight: History augmentation improves success rate under aliasing")
    print("             by using temporal context to disambiguate states.")
    print("=" * 70)


if __name__ == "__main__":
    main()
