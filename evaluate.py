"""Paired evaluation of policies on unseen, simulated market paths.

Protocol and interpretation limits: docs/EVALUATION_PROTOCOL.md.
"""

from collections import defaultdict
from statistics import mean, median, pstdev

import numpy as np

from env import OptionsEnv


POLICIES = ("cash", "random", "regime_call", "ppo")


def choose_action(policy, observation, random_generator, model=None):
    """Choose action using agent-visible observation only."""
    if policy == "cash":
        return 4
    if policy == "random":
        return int(random_generator.integers(0, 5))
    if policy == "regime_call":
        if len(observation) < 15 or observation[14] not in (-1, 1):
            raise ValueError("regime_call requires observed bull/bear regime at index 14")
        return 0 if observation[14] == 1 else 2
    if policy == "ppo":
        if model is None:
            raise ValueError("ppo policy requires a trained model")
        action, _ = model.predict(observation, deterministic=True)
        return int(action)
    raise ValueError(f"Unknown policy: {policy}")


def run_policy_episode(policy, seed, model=None, episode_length=60, training_seed=None):
    """Run one policy on one reproducible simulator path."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    env = OptionsEnv(episode_length=episode_length)
    observation, info = env.reset(seed=seed)
    action_rng = np.random.default_rng(seed + 1_000_000)
    initial_value = float(info["cash"])
    values = [initial_value]
    spots = [float(info["spot"])]
    regimes = [int(info["regime"])]
    actions = []
    total_reward = 0.0
    total_spread = 0.0

    done = False
    while not done:
        action = choose_action(policy, observation, action_rng, model=model)
        observation, reward, terminated, truncated, info = env.step(action)
        values.append(float(info["portfolio_value"]))
        spots.append(float(info["spot"]))
        regimes.append(int(info["regime"]))
        actions.append(action)
        total_reward += float(reward)
        total_spread += float(info["transaction_cost"])
        done = terminated or truncated

    peaks = np.maximum.accumulate(values)
    drawdown = 100 * np.max(1 - np.asarray(values) / peaks)
    return {
        "policy": policy,
        "seed": int(seed),
        "training_seed": training_seed,
        "terminal_return_pct": 100 * (values[-1] / initial_value - 1),
        "max_drawdown_pct": float(drawdown),
        "spread_paid_dollars": total_spread,
        "total_shaped_reward": total_reward,
        "final_portfolio_value": values[-1],
        "portfolio_values": values,
        "spots": spots,
        "regimes": regimes,
        "actions": actions,
    }


def summarize_results(rows, metric="terminal_return_pct"):
    """Summarize one metric without mixing independently trained PPO runs."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row.get("training_seed"))].append(float(row[metric]))
    return {
        key: {
            "n": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "std": pstdev(values),
        }
        for key, values in groups.items()
    }
