"""
Visualization Script: Analyze trained agent behavior.

This script:
1. Loads a trained model
2. Runs episodes and records all data
3. Visualizes agent behavior and performance

Usage:
    python visualize.py
    python visualize.py --model models/ppo_options_XXXX.zip
"""

import argparse
import glob
import os
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from env import OptionsEnv


def get_experiment_dir_from_model(model_path):
    """
    Extract the experiment directory from a model path.
    
    If model is at experiments/20260105_193822_xxx/model.zip,
    returns experiments/20260105_193822_xxx/
    
    Args:
        model_path: Path to model file
        
    Returns:
        Path to experiment directory (parent of model file)
    """
    # Get the directory containing the model
    exp_dir = os.path.dirname(model_path)
    
    # If it's a valid experiment directory (in experiments/), use it
    if "experiments" in exp_dir:
        return exp_dir
    
    # Otherwise create a new one (fallback for old model paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    exp_name = f"{timestamp}_{short_uuid}"
    new_dir = os.path.join("experiments", exp_name)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


# =============================================================================
# Episode Recording
# =============================================================================

def run_episode(env, model, seed=None, deterministic=True):
    """
    Run one episode and record all data.
    
    NEW: Updated for call/put environment with 5 actions.
    
    Returns:
        dict with episode data:
            - observations: list of state vectors
            - actions: list of actions taken
            - rewards: list of rewards
            - infos: list of info dicts
            - total_reward: sum of rewards
    """
    obs, info = env.reset(seed=seed)
    
    # Calculate initial portfolio value
    initial_value = info["cash"]
    
    data = {
        "observations": [obs.copy()],
        "actions": [],
        "rewards": [],
        "infos": [info.copy()],
        "spots": [info["spot"]],
        # NEW: Track both call and put prices
        "call_prices": [info["call_price"]],
        "put_prices": [info["put_price"]],
        # NEW: Track both positions
        "call_positions": [info["call_position"]],
        "put_positions": [info["put_position"]],
        "portfolio_deltas": [info["portfolio_delta"]],
        "portfolio_values": [initial_value],
        "regimes": [info["regime"]],
        "interest_earned": [info.get("total_interest_earned", 0)],
    }
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Calculate portfolio value (cash + positions)
        call_value = info["call_position"] * info["call_price"] * 100
        put_value = info["put_position"] * info["put_price"] * 100
        portfolio_value = info["cash"] + call_value + put_value
        
        data["observations"].append(obs.copy())
        data["actions"].append(int(action))
        data["rewards"].append(reward)
        data["infos"].append(info.copy())
        data["spots"].append(info["spot"])
        data["call_prices"].append(info["call_price"])
        data["put_prices"].append(info["put_price"])
        data["call_positions"].append(info["call_position"])
        data["put_positions"].append(info["put_position"])
        data["portfolio_deltas"].append(info["portfolio_delta"])
        data["portfolio_values"].append(portfolio_value)
        data["regimes"].append(info["regime"])
        data["interest_earned"].append(info.get("total_interest_earned", 0))
    
    data["total_reward"] = sum(data["rewards"])
    return data


def run_random_episode(env, seed=None):
    """Run one episode with random actions."""
    obs, info = env.reset(seed=seed)
    
    initial_value = info["cash"]
    data = {
        "rewards": [],
        "portfolio_values": [initial_value],
    }
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Calculate portfolio value
        call_value = info["call_position"] * info["call_price"] * 100
        put_value = info["put_position"] * info["put_price"] * 100
        portfolio_value = info["cash"] + call_value + put_value
        
        data["rewards"].append(reward)
        data["portfolio_values"].append(portfolio_value)
    
    data["total_reward"] = sum(data["rewards"])
    return data


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_episode_trajectory(data, title="Episode Trajectory"):
    """
    Plot a single episode showing stock, options, positions, and portfolio.
    NEW: Updated for call/put environment.
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    
    steps = range(len(data["spots"]))
    
    # Plot 1: Stock and Option Prices
    ax1 = axes[0]
    ax1.plot(steps, data["spots"], 'b-', linewidth=2, label="Stock Price")
    ax1.set_ylabel("Stock Price ($)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Add regime background
    for i, regime in enumerate(data["regimes"]):
        color = 'lightgreen' if regime == 1 else 'lightcoral'
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=color)
    
    ax1b = ax1.twinx()
    ax1b.plot(steps, data["call_prices"], 'g-', linewidth=2, label="Call Price", alpha=0.8)
    ax1b.plot(steps, data["put_prices"], 'r-', linewidth=2, label="Put Price", alpha=0.8)
    ax1b.set_ylabel("Option Price ($)")
    ax1b.legend(loc='upper right')
    
    ax1.legend(loc='upper left')
    
    # Plot 2: Call and Put Positions
    ax2 = axes[1]
    width = 0.35
    x = np.array(list(steps))
    ax2.bar(x - width/2, data["call_positions"], width, label='Call Position', color='blue', alpha=0.7)
    ax2.bar(x + width/2, data["put_positions"], width, label='Put Position', color='orange', alpha=0.7)
    ax2.set_ylabel("Position")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Short", "Flat", "Long"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Positions (Call=Blue, Put=Orange)")
    
    # Plot 3: Portfolio Delta
    ax3 = axes[2]
    delta_colors = ['green' if d >= 0 else 'red' for d in data["portfolio_deltas"]]
    ax3.bar(steps, data["portfolio_deltas"], color=delta_colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel("Portfolio Delta")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Portfolio Delta (Green=Bullish, Red=Bearish)")
    
    # Plot 4: Rewards
    ax4 = axes[3]
    reward_colors = ['green' if r >= 0 else 'red' for r in data["rewards"]]
    ax4.bar(range(1, len(data["rewards"]) + 1), data["rewards"], color=reward_colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel("Reward")
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Portfolio Value and Interest
    ax5 = axes[4]
    ax5.plot(steps, data["portfolio_values"], 'purple', linewidth=2, label="Portfolio Value")
    ax5.axhline(y=10000, color='gray', linestyle='--', linewidth=1, label="Initial")
    ax5.fill_between(steps, 10000, data["portfolio_values"], 
                     where=[v >= 10000 for v in data["portfolio_values"]], 
                     color='green', alpha=0.3)
    ax5.fill_between(steps, 10000, data["portfolio_values"],
                     where=[v < 10000 for v in data["portfolio_values"]],
                     color='red', alpha=0.3)
    ax5.set_ylabel("Portfolio Value ($)")
    ax5.set_xlabel("Step (Day)")
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Add interest earned annotation
    final_interest = data["interest_earned"][-1] if data["interest_earned"] else 0
    ax5.annotate(f'Interest Earned: ${final_interest:.2f}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, color='blue')
    
    plt.tight_layout()
    return fig


def plot_action_distribution(episodes_data):
    """
    Plot distribution of actions taken by the agent.
    NEW: Updated for 5-action environment (calls + puts).
    """
    all_actions = []
    for data in episodes_data:
        all_actions.extend(data["actions"])
    
    action_counts = [
        all_actions.count(0),  # BUY_CALL
        all_actions.count(1),  # BUY_PUT
        all_actions.count(2),  # SELL_CALL
        all_actions.count(3),  # SELL_PUT
        all_actions.count(4),  # HOLD
    ]
    action_labels = ["BUY\nCALL", "BUY\nPUT", "SELL\nCALL", "SELL\nPUT", "HOLD"]
    colors = ['limegreen', 'lightgreen', 'red', 'lightcoral', 'gray']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(action_labels, action_counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add count labels on bars
    total = len(all_actions) if len(all_actions) > 0 else 1
    for bar, count in zip(bars, action_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{count}\n({count/total*100:.1f}%)', 
               ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution (Call=Green tones, Put=Red tones)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    call_actions = action_counts[0] + action_counts[2]  # BUY_CALL + SELL_CALL
    put_actions = action_counts[1] + action_counts[3]   # BUY_PUT + SELL_PUT
    hold_actions = action_counts[4]
    
    summary = f"Calls: {call_actions} ({call_actions/total*100:.1f}%) | "
    summary += f"Puts: {put_actions} ({put_actions/total*100:.1f}%) | "
    summary += f"Hold: {hold_actions} ({hold_actions/total*100:.1f}%)"
    ax.set_xlabel(summary)
    
    plt.tight_layout()
    return fig


def plot_reward_comparison(trained_rewards, random_rewards):
    """
    Compare trained agent vs random agent rewards.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(trained_rewards, bins=15, alpha=0.7, label='Trained Agent', color='blue', edgecolor='black')
    ax1.hist(random_rewards, bins=15, alpha=0.7, label='Random Agent', color='gray', edgecolor='black')
    ax1.axvline(np.mean(trained_rewards), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(random_rewards), color='gray', linestyle='--', linewidth=2)
    ax1.set_xlabel("Episode Reward")
    ax1.set_ylabel("Count")
    ax1.set_title("Reward Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([trained_rewards, random_rewards], 
                     labels=['Trained', 'Random'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgray')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel("Episode Reward")
    ax2.set_title("Reward Comparison")
    ax2.grid(True, alpha=0.3)
    
    # Add means as text
    ax2.text(1, np.mean(trained_rewards), f'μ={np.mean(trained_rewards):.3f}', 
            ha='center', va='bottom', fontsize=10, color='blue')
    ax2.text(2, np.mean(random_rewards), f'μ={np.mean(random_rewards):.3f}',
            ha='center', va='bottom', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig


def plot_greeks_vs_actions(episodes_data):
    """
    Scatter plot: How do Greeks relate to actions?
    """
    deltas = []
    gammas = []
    actions = []
    
    for data in episodes_data:
        for i, obs in enumerate(data["observations"][:-1]):  # Exclude last (no action)
            deltas.append(obs[2])  # delta is index 2
            gammas.append(obs[3])  # gamma is index 3
            actions.append(data["actions"][i])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    action_labels = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
    action_colors = {0: 'green', 1: 'gray', 2: 'red'}
    
    for action in [0, 1, 2]:
        mask = [a == action for a in actions]
        ax.scatter(
            [d for d, m in zip(deltas, mask) if m],
            [g for g, m in zip(gammas, mask) if m],
            c=action_colors[action],
            label=action_labels[action],
            alpha=0.5,
            s=50
        )
    
    ax.set_xlabel("Delta")
    ax.set_ylabel("Gamma")
    ax.set_title("Agent Actions by Delta and Gamma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main(model_path=None, n_episodes=20, seed=42):
    """
    Load model and generate visualizations.
    """
    print("=" * 60)
    print("OPTIONS RL VISUALIZATION")
    print("=" * 60)
    
    # Find latest model if not specified
    if model_path is None:
        # Look in experiments directory first
        model_files = glob.glob("experiments/*/model.zip")
        if not model_files:
            # Fallback to old models directory
            model_files = glob.glob("models/ppo_options_*.zip")
        if not model_files:
            print("No trained models found.")
            print("Run train.py first to train a model.")
            return
        model_path = max(model_files, key=os.path.getctime)
        print(f"\nUsing latest model: {model_path}")
    
    # Load model and environment
    print("\n1. Loading model and environment...")
    env = OptionsEnv()
    model = PPO.load(model_path, env=env)
    print(f"   Model loaded from: {model_path}")
    
    # Run episodes with trained agent
    print(f"\n2. Running {n_episodes} episodes with trained agent...")
    trained_episodes = []
    for i in range(n_episodes):
        data = run_episode(env, model, seed=seed + i, deterministic=True)
        trained_episodes.append(data)
        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/{n_episodes} episodes")
    
    trained_rewards = [d["total_reward"] for d in trained_episodes]
    print(f"   Avg Reward: {np.mean(trained_rewards):+.4f}")
    
    # Run episodes with random agent
    print(f"\n3. Running {n_episodes} episodes with random agent...")
    random_episodes = []
    for i in range(n_episodes):
        data = run_random_episode(env, seed=seed + i)
        random_episodes.append(data)
    
    random_rewards = [d["total_reward"] for d in random_episodes]
    print(f"   Avg Reward: {np.mean(random_rewards):+.4f}")
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    
    # Use the same experiment directory as the model
    exp_dir = get_experiment_dir_from_model(model_path)
    print(f"   Saving to model's experiment directory: {exp_dir}")
    
    # Plot 1: Single episode trajectory
    fig1 = plot_episode_trajectory(trained_episodes[0], "Trained Agent - Episode 1")
    fig1.savefig(os.path.join(exp_dir, "episode_trajectory.png"), dpi=150, bbox_inches='tight')
    print(f"   Saved: {exp_dir}/episode_trajectory.png")
    
    # Plot 2: Action distribution
    fig2 = plot_action_distribution(trained_episodes)
    fig2.savefig(os.path.join(exp_dir, "action_distribution.png"), dpi=150, bbox_inches='tight')
    print(f"   Saved: {exp_dir}/action_distribution.png")
    
    # Plot 3: Reward comparison
    fig3 = plot_reward_comparison(trained_rewards, random_rewards)
    fig3.savefig(os.path.join(exp_dir, "reward_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"   Saved: {exp_dir}/reward_comparison.png")
    
    # Plot 4: Greeks vs Actions
    fig4 = plot_greeks_vs_actions(trained_episodes)
    fig4.savefig(os.path.join(exp_dir, "greeks_vs_actions.png"), dpi=150, bbox_inches='tight')
    print(f"   Saved: {exp_dir}/greeks_vs_actions.png")
    
    # Save visualization metadata (append to existing or create new)
    metadata_path = os.path.join(exp_dir, "visualization_metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Visualization run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Trained Avg Reward: {np.mean(trained_rewards):.4f}\n")
        f.write(f"Random Avg Reward: {np.mean(random_rewards):.4f}\n")
        f.write(f"Improvement: {np.mean(trained_rewards) - np.mean(random_rewards):.4f}\n")
    print(f"   Saved: {exp_dir}/visualization_metadata.txt")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTrained Agent:")
    print(f"  Mean Reward:   {np.mean(trained_rewards):+.4f}")
    print(f"  Std Reward:    {np.std(trained_rewards):.4f}")
    print(f"  Min Reward:    {np.min(trained_rewards):+.4f}")
    print(f"  Max Reward:    {np.max(trained_rewards):+.4f}")
    
    print(f"\nRandom Agent:")
    print(f"  Mean Reward:   {np.mean(random_rewards):+.4f}")
    print(f"  Std Reward:    {np.std(random_rewards):.4f}")
    
    print(f"\nImprovement: {np.mean(trained_rewards) - np.mean(random_rewards):+.4f}")
    
    print("\n" + "=" * 60)
    print(f"Experiment saved to: {exp_dir}/")
    print("=" * 60)
    
    return exp_dir
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained RL agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (default: latest in models/)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to run (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    main(model_path=args.model, n_episodes=args.episodes, seed=args.seed)
