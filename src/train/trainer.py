"""Training script for RL scheduling agents.

This script provides comprehensive training functionality with proper logging,
evaluation, and checkpointing for different RL algorithms.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.envs.scheduling_env import JobSchedulingEnv
from src.algorithms.dqn import DQNAgent
from src.algorithms.ppo import PPOAgent
from src.utils.logging import Logger
from src.utils.metrics import MetricsTracker


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    env_name: str = "JobScheduling"
    num_resources: int = 3
    max_jobs: int = 10
    job_arrival_rate: float = 0.3
    max_duration: float = 10.0
    max_priority: float = 5.0
    time_horizon: float = 100.0
    
    # Algorithm
    algorithm: str = "dqn"  # "dqn", "ppo"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    
    # DQN specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    use_double_dqn: bool = True
    use_noisy: bool = False
    use_per: bool = False
    
    # PPO specific
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    
    # Training
    num_episodes: int = 10000
    eval_freq: int = 100
    eval_episodes: int = 10
    save_freq: int = 1000
    log_freq: int = 10
    
    # Logging
    use_wandb: bool = False
    project_name: str = "rl-scheduling"
    experiment_name: str = "default"
    
    # Device
    device: str = "auto"
    
    # Reproducibility
    seed: int = 42


class Trainer:
    """Main trainer class for RL scheduling agents."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create output directory
        self.output_dir = Path(f"outputs/{config.experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        # Initialize logging
        self.logger = Logger(self.output_dir / "training.log")
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=asdict(config),
                dir=str(self.output_dir)
            )
        
        # Create environment
        self.env = JobSchedulingEnv(
            num_resources=config.num_resources,
            max_jobs=config.max_jobs,
            job_arrival_rate=config.job_arrival_rate,
            max_duration=config.max_duration,
            max_priority=config.max_priority,
            time_horizon=config.time_horizon,
            seed=config.seed
        )
        
        # Create agent
        self.agent = self._create_agent()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
        self.logger.info(f"Initialized trainer with config: {config}")
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_agent(self):
        """Create RL agent based on config."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        if self.config.algorithm == "dqn":
            return DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                target_update_freq=self.config.target_update_freq,
                device=self.config.device,
                use_double_dqn=self.config.use_double_dqn,
                use_noisy=self.config.use_noisy,
                use_per=self.config.use_per,
            )
        elif self.config.algorithm == "ppo":
            return PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                lam=self.config.lam,
                clip_ratio=self.config.clip_ratio,
                value_loss_coef=self.config.value_loss_coef,
                entropy_coef=self.config.entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
                ppo_epochs=self.config.ppo_epochs,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            if self.config.algorithm == "dqn":
                action = self.agent.select_action(state, training=True)
                
                # Take step
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train step
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
            elif self.config.algorithm == "ppo":
                action, log_prob, value = self.agent.select_action(state, training=True)
                
                # Take step
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                # Store experience
                self.agent.store_experience(state, action, reward, value, log_prob)
                
                # Update if buffer is full
                if self.agent.buffer.size >= self.agent.buffer.buffer_size:
                    metrics = self.agent.update(next_state)
                    if metrics:
                        episode_losses.append(metrics.get('policy_loss', 0.0))
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            state = next_state
        
        # Final update for PPO
        if self.config.algorithm == "ppo" and self.agent.buffer.size > 0:
            metrics = self.agent.update(next_state)
            if metrics:
                episode_losses.append(metrics.get('policy_loss', 0.0))
        
        # Calculate episode metrics
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'avg_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'env_metrics': info.get('metrics', {}),
        }
        
        return episode_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        eval_rewards = []
        eval_metrics = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            
            episode_reward = 0.0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action (no exploration)
                if self.config.algorithm == "dqn":
                    action = self.agent.select_action(state, training=False)
                elif self.config.algorithm == "ppo":
                    action, _, _ = self.agent.select_action(state, training=False)
                
                # Take step
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_metrics.append(info.get('metrics', {}))
        
        # Calculate evaluation metrics
        eval_results = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_reward_min': np.min(eval_rewards),
            'eval_reward_max': np.max(eval_rewards),
        }
        
        # Aggregate environment metrics
        if eval_metrics:
            for key in eval_metrics[0].keys():
                values = [m[key] for m in eval_metrics if key in m]
                eval_results[f'eval_{key}_mean'] = np.mean(values)
                eval_results[f'eval_{key}_std'] = np.std(values)
        
        return eval_results
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        start_time = time.time()
        
        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            self.episode = episode
            
            # Train episode
            episode_metrics = self.train_episode()
            
            # Update metrics tracker
            self.metrics_tracker.update(episode_metrics)
            
            # Logging
            if episode % self.config.log_freq == 0:
                self._log_episode(episode, episode_metrics)
            
            # Evaluation
            if episode % self.config.eval_freq == 0:
                eval_results = self.evaluate(self.config.eval_episodes)
                self._log_evaluation(episode, eval_results)
                
                # Save best model
                if eval_results['eval_reward_mean'] > self.best_eval_reward:
                    self.best_eval_reward = eval_results['eval_reward_mean']
                    self.save_checkpoint("best_model.pt")
            
            # Save checkpoint
            if episode % self.config.save_freq == 0:
                self.save_checkpoint(f"checkpoint_episode_{episode}.pt")
        
        # Final evaluation and save
        final_eval = self.evaluate(50)  # More episodes for final evaluation
        self._log_evaluation(self.config.num_episodes, final_eval)
        self.save_checkpoint("final_model.pt")
        
        # Save training curves
        self._save_training_curves()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        if self.config.use_wandb:
            wandb.finish()
    
    def _log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log episode metrics."""
        log_msg = f"Episode {episode}: Reward={metrics['episode_reward']:.2f}, "
        log_msg += f"Steps={metrics['episode_steps']}, Loss={metrics['avg_loss']:.4f}"
        
        self.logger.info(log_msg)
        
        if self.config.use_wandb:
            wandb.log({
                'episode': episode,
                'episode_reward': metrics['episode_reward'],
                'episode_steps': metrics['episode_steps'],
                'avg_loss': metrics['avg_loss'],
                **metrics['env_metrics']
            })
    
    def _log_evaluation(self, episode: int, eval_results: Dict[str, float]) -> None:
        """Log evaluation results."""
        log_msg = f"Evaluation at episode {episode}: "
        log_msg += f"Reward={eval_results['eval_reward_mean']:.2f}±{eval_results['eval_reward_std']:.2f}"
        
        self.logger.info(log_msg)
        
        if self.config.use_wandb:
            wandb.log({
                'episode': episode,
                **eval_results
            })
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        self.agent.save(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_eval_reward': self.best_eval_reward,
            'config': asdict(self.config)
        }
        
        with open(self.output_dir / f"{filename}.json", "w") as f:
            json.dump(training_state, f, indent=2)
    
    def _save_training_curves(self) -> None:
        """Save training curves."""
        metrics = self.metrics_tracker.get_all_metrics()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(metrics['episode_reward'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode steps
        axes[0, 1].plot(metrics['episode_steps'])
        axes[0, 1].set_title('Episode Steps')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Loss
        if metrics['avg_loss']:
            axes[1, 0].plot(metrics['avg_loss'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
        
        # Resource utilization
        if 'resource_utilization' in metrics:
            axes[1, 1].plot(metrics['resource_utilization'])
            axes[1, 1].set_title('Resource Utilization')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Utilization')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL scheduling agent")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--algorithm", type=str, choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line args
    config.algorithm = args.algorithm
    config.num_episodes = args.num_episodes
    config.experiment_name = args.experiment_name
    config.use_wandb = args.use_wandb
    config.seed = args.seed
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
