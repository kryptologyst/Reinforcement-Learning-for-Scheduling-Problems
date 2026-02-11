"""Evaluation script for trained models."""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from envs.scheduling_env import JobSchedulingEnv
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent


def load_and_evaluate(model_path: str, config_path: str, algorithm: str, num_episodes: int = 100):
    """Load model and evaluate performance."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = JobSchedulingEnv(
        num_resources=config['env']['num_resources'],
        max_jobs=config['env']['max_jobs'],
        job_arrival_rate=config['env']['job_arrival_rate'],
        max_duration=config['env']['max_duration'],
        max_priority=config['env']['max_priority'],
        time_horizon=config['env']['time_horizon'],
        seed=config['seed']
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            epsilon_start=config['dqn']['epsilon_start'],
            epsilon_end=config['dqn']['epsilon_end'],
            epsilon_decay=config['dqn']['epsilon_decay'],
            buffer_size=config['dqn']['buffer_size'],
            batch_size=config['dqn']['batch_size'],
            target_update_freq=config['dqn']['target_update_freq'],
            device="cpu",  # Use CPU for evaluation
            use_double_dqn=config['dqn']['use_double_dqn'],
            use_noisy=config['dqn']['use_noisy'],
            use_per=config['dqn']['use_per'],
        )
    elif algorithm == "ppo":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            lam=config['ppo']['lam'],
            clip_ratio=config['ppo']['clip_ratio'],
            value_loss_coef=config['ppo']['value_loss_coef'],
            entropy_coef=config['ppo']['entropy_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            ppo_epochs=config['ppo']['ppo_epochs'],
            buffer_size=config['ppo']['buffer_size'],
            batch_size=config['ppo']['batch_size'],
            device="cpu",  # Use CPU for evaluation
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load trained weights
    agent.load(model_path)
    
    # Evaluate
    rewards = []
    metrics = []
    
    print(f"Evaluating {algorithm.upper()} agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        episode_reward = 0.0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (no exploration)
            if algorithm == "dqn":
                action = agent.select_action(state, training=False)
            elif algorithm == "ppo":
                action, _, _ = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        metrics.append(info.get('metrics', {}))
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")
    
    # Calculate statistics
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    reward_min = np.min(rewards)
    reward_max = np.max(rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}")
    print(f"Min Reward: {reward_min:.2f}")
    print(f"Max Reward: {reward_max:.2f}")
    
    # Aggregate environment metrics
    if metrics:
        print(f"\nEnvironment Metrics:")
        for key in metrics[0].keys():
            values = [m[key] for m in metrics if key in m]
            print(f"{key}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    return {
        'rewards': rewards,
        'metrics': metrics,
        'statistics': {
            'mean': reward_mean,
            'std': reward_std,
            'min': reward_min,
            'max': reward_max
        }
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL scheduling agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--algorithm", type=str, choices=["dqn", "ppo"], required=True, help="Algorithm type")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        return
    
    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} not found")
        return
    
    # Run evaluation
    try:
        results = load_and_evaluate(args.model, args.config, args.algorithm, args.episodes)
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
