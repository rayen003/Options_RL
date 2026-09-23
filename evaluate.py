"""Paired evaluation of policies on unseen, simulated market paths.

Protocol and interpretation limits: docs/EVALUATION_PROTOCOL.md.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from env import OptionsEnv


POLICIES = ("cash", "random", "regime_call", "ppo")
METRICS = (
    "terminal_return_pct",
    "max_drawdown_pct",
    "spread_paid_dollars",
    "total_shaped_reward",
)


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


def run_study(seeds=range(10_000, 10_020), models=None, episode_length=60):
    """Run three fixed baselines and every supplied PPO model on matched paths."""
    models = models or {}
    rows = []
    for seed in seeds:
        for policy in POLICIES[:3]:
            rows.append(run_policy_episode(policy, seed, episode_length=episode_length))
        for training_seed, model in sorted(models.items()):
            rows.append(
                run_policy_episode(
                    "ppo",
                    seed,
                    model=model,
                    episode_length=episode_length,
                    training_seed=training_seed,
                )
            )
    return rows


def paired_differences(rows):
    """Subtract cash/random return on same simulated market seed."""
    references = {
        (row["seed"], row["policy"]): row["terminal_return_pct"]
        for row in rows
        if row["policy"] in ("cash", "random")
    }
    paired = []
    for row in rows:
        if row["policy"] not in ("regime_call", "ppo"):
            continue
        seed = row["seed"]
        if (seed, "cash") not in references or (seed, "random") not in references:
            raise ValueError(f"Missing paired cash/random result for seed {seed}")
        paired.append(
            {
                "policy": row["policy"],
                "training_seed": row["training_seed"],
                "seed": seed,
                "terminal_return_pct": row["terminal_return_pct"],
                "vs_random_pct_points": row["terminal_return_pct"] - references[(seed, "random")],
                "vs_cash_pct_points": row["terminal_return_pct"] - references[(seed, "cash")],
            }
        )
    return paired


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def _policy_label(policy, training_seed):
    return f"PPO {training_seed}" if policy == "ppo" else policy.replace("_", " ")


def write_report(rows, output_dir, episode_length=60, model_paths=None):
    """Save raw outcomes, paired comparisons, summary, and labeled plot."""
    if not rows:
        raise ValueError("Cannot report zero episodes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in (
        "episodes.csv", "summary.csv", "paired_differences.csv", "comparison.png", "manifest.json"
    )]
    if any(target.exists() for target in targets):
        raise FileExistsError(f"Evaluation report already exists in {output_dir}")

    episode_columns = (
        "policy", "training_seed", "seed", "terminal_return_pct", "max_drawdown_pct",
        "spread_paid_dollars", "total_shaped_reward", "final_portfolio_value"
    )
    _write_csv(targets[0], rows, episode_columns)
    paired = paired_differences(rows)
    _write_csv(
        targets[2], paired,
        ("policy", "training_seed", "seed", "terminal_return_pct", "vs_random_pct_points", "vs_cash_pct_points"),
    )

    summary_rows = []
    for metric in METRICS:
        for (policy, training_seed), values in summarize_results(rows, metric).items():
            summary_rows.append({"policy": policy, "training_seed": training_seed, "metric": metric, **values})
    _write_csv(
        targets[1], summary_rows,
        ("policy", "training_seed", "metric", "n", "mean", "median", "min", "max", "std"),
    )

    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["training_seed"])].append(row["terminal_return_pct"])
    labels = [_policy_label(*key) for key in groups]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(list(groups.values()), labels=labels)
    axes[0].set_ylabel("Terminal portfolio return (%)")
    axes[0].set_title("Same unseen simulated paths")
    paired_groups = defaultdict(list)
    for row in paired:
        paired_groups[(row["policy"], row["training_seed"])].append(row["vs_random_pct_points"])
    if paired_groups:
        axes[1].boxplot(
            list(paired_groups.values()),
            labels=[_policy_label(*key) for key in paired_groups],
        )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Return minus random (percentage points)")
    axes[1].set_title("Paired difference on each seed")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.01, "Synthetic simulator results; not evidence of real-market performance.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(targets[3], dpi=150)
    plt.close(fig)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, check=False,
    )
    manifest = {
        "protocol": "docs/EVALUATION_PROTOCOL.md",
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "evaluation_seeds": sorted({int(row["seed"]) for row in rows}),
        "training_seeds": sorted({int(row["training_seed"]) for row in rows if row["training_seed"] is not None}),
        "episode_length": episode_length,
        "model_paths": {str(seed): str(path) for seed, path in (model_paths or {}).items()},
        "warning": "Results come from simulated BSM quotes and paths, not real-market performance.",
    }
    targets[4].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare policies on matched simulated options paths")
    parser.add_argument("--episodes", type=int, default=20, help="Number of held-out paths (default: 20)")
    parser.add_argument("--seed-start", type=int, default=10_000, help="First path seed (default: 10000)")
    parser.add_argument("--episode-length", type=int, default=60, help="Trading days per path")
    parser.add_argument("--output", type=Path, required=True, help="New directory for report files")
    parser.add_argument(
        "--model", action="append", default=[], metavar="SEED=PATH",
        help="PPO training seed and model .zip; repeat for each independent run",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.episode_length <= 0 or args.seed_start < 0:
        parser.error("episodes and episode-length must be positive; seed-start must be nonnegative")

    model_paths = {}
    for spec in args.model:
        seed_text, separator, path = spec.partition("=")
        if not separator or not path:
            parser.error(f"Invalid --model '{spec}'; expected SEED=PATH")
        try:
            training_seed = int(seed_text)
        except ValueError:
            parser.error(f"Invalid training seed in --model '{spec}'")
        if training_seed in model_paths:
            parser.error(f"Duplicate training seed {training_seed}")
        model_paths[training_seed] = Path(path)

    models = {}
    if model_paths:
        from stable_baselines3 import PPO

        for training_seed, path in model_paths.items():
            models[training_seed] = PPO.load(path)

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    rows = run_study(seeds=seeds, models=models, episode_length=args.episode_length)
    write_report(rows, args.output, episode_length=args.episode_length, model_paths=model_paths)
    print("Simulated options-path evaluation; not real-market performance.")
    for (policy, training_seed), stats in summarize_results(rows).items():
        label = _policy_label(policy, training_seed)
        print(f"{label}: median terminal return {stats['median']:+.2f}% across {stats['n']} paths")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
