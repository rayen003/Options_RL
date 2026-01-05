"""
Training Script: Train an RL agent to trade options.

This script uses Stable-Baselines3's PPO algorithm to train an agent
on our OptionsEnv environment.

Usage:
    python train.py                    # Train for default steps
    python train.py --timesteps 50000  # Train for 50k steps
"""

import argparse
import os
import uuid
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from env import OptionsEnv


def create_experiment_dir(base_dir="experiments"):
    """
    Create a unique experiment directory with datetime + uuid.
    
    Format: experiments/YYYYMMDD_HHMMSS_<short_uuid>/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    exp_name = f"{timestamp}_{short_uuid}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# =============================================================================
# Custom Callback for Training Progress
# =============================================================================

class TrainingCallback(BaseCallback):
    """
    Custom callback to track and display training progress.
    
    Shows:
        - Episode rewards
        - Average reward over last 10 episodes
        - Best episode so far
    """
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_reward = -np.inf
    
    def _on_step(self) -> bool:
        # Accumulate rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Update best
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
            
            # Print progress every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"  Episode {len(self.episode_rewards):4d} | "
                      f"Avg Reward (10ep): {avg_reward:+.4f} | "
                      f"Best: {self.best_reward:+.4f}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


# =============================================================================
# Training Function
# =============================================================================

def train(
    total_timesteps: int = 20_000,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    episode_length: int = 60,
):
    """
    Train a PPO agent on the options trading environment.
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for the optimizer
        n_steps: Steps per environment per update
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs per update
        gamma: Discount factor
        seed: Random seed
        episode_length: Number of trading days per episode (default: 60)
    
    Returns:
        model: Trained PPO model
        episode_rewards: List of rewards per episode
        exp_dir: Path to experiment directory
    """
    print("=" * 60)
    print("OPTIONS RL TRAINING")
    print("=" * 60)
    
    # =========================================================================
    # Create Environment
    # =========================================================================
    print("\n1. Creating Environment...")
    
    env = OptionsEnv(seed=seed, episode_length=episode_length)
    
    print(f"   Observation Space: {env.observation_space.shape}")
    print(f"   Action Space: {env.action_space.n} actions")
    print(f"   Episode Length: {env.episode_length} trading days (~{env.episode_length/21:.1f} months)")
    
    # =========================================================================
    # Create PPO Agent
    # =========================================================================
    print("\n2. Creating PPO Agent...")
    
    model = PPO(
        policy="MlpPolicy",           # Multi-layer perceptron policy
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,              # Steps to collect before update
        batch_size=batch_size,        # Minibatch size
        n_epochs=n_epochs,            # Epochs per update
        gamma=gamma,                  # Discount factor
        verbose=0,                    # Suppress SB3 output
        seed=seed,
        tensorboard_log=None,         # Disable tensorboard for now
    )
    
    print(f"   Policy: MlpPolicy (2 hidden layers, 64 units each)")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Gamma (discount): {gamma}")
    print(f"   Steps per Update: {n_steps}")
    print(f"   Batch Size: {batch_size}")
    
    # =========================================================================
    # Train
    # =========================================================================
    print(f"\n3. Training for {total_timesteps:,} timesteps...")
    print("-" * 60)
    
    callback = TrainingCallback()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    print("-" * 60)
    print(f"   Training complete!")
    print(f"   Total Episodes: {len(callback.episode_rewards)}")
    print(f"   Best Episode Reward: {callback.best_reward:+.4f}")
    print(f"   Final Avg Reward (10ep): {np.mean(callback.episode_rewards[-10:]):+.4f}")
    
    # =========================================================================
    # Save Model and Experiment
    # =========================================================================
    print("\n4. Saving Model...")
    
    # Create experiment directory
    exp_dir = create_experiment_dir("experiments")
    model_path = os.path.join(exp_dir, "model")
    model.save(model_path)
    
    print(f"   Experiment directory: {exp_dir}")
    print(f"   Model saved to: {model_path}.zip")
    
    # Save training metadata
    metadata_path = os.path.join(exp_dir, "training_metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Episode Length: {episode_length} trading days\n")
        f.write(f"Total Episodes: {len(callback.episode_rewards)}\n")
        f.write(f"Best Episode Reward: {callback.best_reward:.4f}\n")
        f.write(f"Final Avg Reward (10ep): {np.mean(callback.episode_rewards[-10:]):.4f}\n")
    print(f"   Metadata saved to: {metadata_path}")
    
    # =========================================================================
    # Quick Evaluation
    # =========================================================================
    print("\n5. Quick Evaluation (5 episodes)...")
    print("-" * 60)
    
    eval_rewards = []
    for ep in range(5):
        obs, info = env.reset(seed=seed + ep + 1000)
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        print(f"   Episode {ep+1}: Reward = {episode_reward:+.4f}, "
              f"Final Portfolio = ${info['portfolio_value']:.2f}")
    
    print("-" * 60)
    print(f"   Avg Eval Reward: {np.mean(eval_rewards):+.4f}")
    
    # =========================================================================
    # Compare to Random Agent
    # =========================================================================
    print("\n6. Comparison: Trained Agent vs Random Agent...")
    print("-" * 60)
    
    random_rewards = []
    for ep in range(5):
        obs, info = env.reset(seed=seed + ep + 2000)
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_rewards.append(episode_reward)
    
    print(f"   Trained Agent Avg Reward: {np.mean(eval_rewards):+.4f}")
    print(f"   Random Agent Avg Reward:  {np.mean(random_rewards):+.4f}")
    print(f"   Improvement: {np.mean(eval_rewards) - np.mean(random_rewards):+.4f}")
    
    # Save reward curve data
    rewards_path = os.path.join(exp_dir, "episode_rewards.npy")
    np.save(rewards_path, np.array(callback.episode_rewards))
    print(f"   Rewards saved to: {rewards_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Experiment saved to: {exp_dir}")
    print("=" * 60)
    
    return model, callback.episode_rewards, exp_dir


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for options trading")
    parser.add_argument("--timesteps", type=int, default=20_000,
                        help="Total training timesteps (default: 20000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--episode-length", type=int, default=60,
                        help="Episode length in trading days (default: 60)")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        episode_length=args.episode_length,
    )
