"""Simple example demonstrating the RL scheduling project.

This script shows how to train and evaluate a basic DQN agent on the job scheduling problem.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from envs.scheduling_env import JobSchedulingEnv
from algorithms.dqn import DQNAgent


def train_simple_agent():
    """Train a simple DQN agent."""
    print("Creating job scheduling environment...")
    
    # Create environment
    env = JobSchedulingEnv(
        num_resources=3,
        max_jobs=5,
        job_arrival_rate=0.3,
        max_duration=8.0,
        max_priority=4.0,
        time_horizon=50.0,
        seed=42
    )
    
    print(f"Environment created:")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.n}")
    print(f"  Number of resources: {env.num_resources}")
    print(f"  Max jobs: {env.max_jobs}")
    
    # Create DQN agent
    print("\nCreating DQN agent...")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        device="cpu"  # Use CPU for simplicity
    )
    
    print("Agent created successfully!")
    
    # Training loop
    print("\nStarting training...")
    num_episodes = 1000
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        episode_reward = 0.0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < 50:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train step
            loss = agent.train_step()
            
            episode_reward += reward
            step_count += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}: Average reward = {avg_reward:.2f}")
    
    print("Training completed!")
    
    # Final evaluation
    print("\nEvaluating trained agent...")
    eval_rewards = []
    
    for eval_episode in range(10):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        episode_reward = 0.0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Average reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Best reward: {np.max(eval_rewards):.2f}")
    print(f"  Worst reward: {np.min(eval_rewards):.2f}")
    
    # Show some episode details
    print(f"\nSample Episode Details:")
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    print(f"  Initial jobs in queue: {len(env.jobs)}")
    print(f"  Resource utilization: {info.get('metrics', {}).get('resource_utilization', 0.0):.2f}")
    
    return agent, env


def demonstrate_agent_behavior(agent, env):
    """Demonstrate agent behavior step by step."""
    print("\n" + "="*50)
    print("AGENT BEHAVIOR DEMONSTRATION")
    print("="*50)
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    print(f"Initial state:")
    print(f"  Jobs in queue: {len(env.jobs)}")
    print(f"  Resources: {len(env.resources)}")
    
    step = 0
    done = False
    truncated = False
    
    while not (done or truncated) and step < 10:
        print(f"\nStep {step + 1}:")
        
        # Show current jobs
        if env.jobs:
            print(f"  Jobs in queue: {len(env.jobs)}")
            for i, job in enumerate(env.jobs[:3]):  # Show first 3 jobs
                print(f"    Job {i}: duration={job.duration:.1f}, priority={job.priority:.1f}")
        else:
            print("  No jobs in queue")
        
        # Show resource status
        print(f"  Resource status:")
        for resource in env.resources:
            status = "busy" if resource.current_job is not None else "idle"
            print(f"    Resource {resource.id}: {status} (capability={resource.capability:.2f})")
        
        # Agent decision
        action = agent.select_action(state, training=False)
        print(f"  Agent chooses action: {action}")
        
        if action == 0:
            print("  Agent decides to wait")
        else:
            job_idx = action - 1
            if job_idx < len(env.jobs):
                job = env.jobs[job_idx]
                print(f"  Agent schedules job {job_idx} (duration={job.duration:.1f}, priority={job.priority:.1f})")
            else:
                print("  Invalid action (no job at that index)")
        
        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Resource utilization: {info.get('metrics', {}).get('resource_utilization', 0.0):.2f}")
        
        state = next_state
        step += 1
    
    print(f"\nEpisode completed after {step} steps")


def main():
    """Main function."""
    print("RL Scheduling Project Demo")
    print("=" * 40)
    
    try:
        # Train agent
        agent, env = train_simple_agent()
        
        # Demonstrate behavior
        demonstrate_agent_behavior(agent, env)
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run 'python scripts/train.py --algorithm dqn --num_episodes 5000' for longer training")
        print("2. Run 'streamlit run demo/app.py' for interactive visualization")
        print("3. Check the README.md for more advanced usage examples")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
