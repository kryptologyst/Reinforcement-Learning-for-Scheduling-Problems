"""Unit tests for the RL scheduling project."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.envs.scheduling_env import JobSchedulingEnv, Job, Resource
from src.algorithms.dqn import DQNAgent, DQNNetwork, ReplayBuffer
from src.algorithms.ppo import PPOAgent, PolicyNetwork, ValueNetwork


class TestSchedulingEnv:
    """Test cases for the scheduling environment."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        env = JobSchedulingEnv(num_resources=3, max_jobs=5, seed=42)
        
        assert env.num_resources == 3
        assert env.max_jobs == 5
        assert env.action_space.n == 6  # max_jobs + 1 (wait action)
        assert env.observation_space.shape[0] > 0
    
    def test_env_reset(self):
        """Test environment reset."""
        env = JobSchedulingEnv(num_resources=2, max_jobs=3, seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == env.observation_space.shape[0]
        assert isinstance(info, dict)
    
    def test_env_step(self):
        """Test environment step."""
        env = JobSchedulingEnv(num_resources=2, max_jobs=3, seed=42)
        obs, info = env.reset()
        
        # Test valid action
        next_obs, reward, done, truncated, info = env.step(0)  # Wait action
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_job_creation(self):
        """Test job creation."""
        job = Job(
            id=0,
            duration=5.0,
            priority=3.0,
            resource_requirements=[0, 1],
            deadline=10.0,
            arrival_time=0.0
        )
        
        assert job.id == 0
        assert job.duration == 5.0
        assert job.priority == 3.0
        assert job.resource_requirements == [0, 1]
        assert job.deadline == 10.0
        assert job.arrival_time == 0.0
    
    def test_resource_creation(self):
        """Test resource creation."""
        resource = Resource(
            id=0,
            capability=1.0,
            current_job=None,
            completion_time=0.0,
            utilization=0.0
        )
        
        assert resource.id == 0
        assert resource.capability == 1.0
        assert resource.current_job is None
        assert resource.completion_time == 0.0
        assert resource.utilization == 0.0


class TestDQN:
    """Test cases for DQN algorithm."""
    
    def test_dqn_network(self):
        """Test DQN network."""
        network = DQNNetwork(state_dim=10, action_dim=5)
        
        # Test forward pass
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization."""
        agent = DQNAgent(state_dim=10, action_dim=5, device="cpu")
        
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.device.type == "cpu"
        assert agent.gamma == 0.99
    
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        agent = DQNAgent(state_dim=10, action_dim=5, device="cpu")
        state = torch.randn(1, 10)
        
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
    
    def test_replay_buffer(self):
        """Test replay buffer."""
        buffer = ReplayBuffer(capacity=100)
        
        # Test empty buffer
        assert len(buffer) == 0
        
        # Test adding transitions
        from src.algorithms.dqn import Transition
        
        transition = Transition(
            state=torch.randn(10),
            action=0,
            reward=1.0,
            next_state=torch.randn(10),
            done=False
        )
        
        buffer.push(transition)
        assert len(buffer) == 1
        
        # Test sampling
        batch = buffer.sample(1)
        assert len(batch) == 1
        assert isinstance(batch[0], Transition)


class TestPPO:
    """Test cases for PPO algorithm."""
    
    def test_policy_network(self):
        """Test policy network."""
        network = PolicyNetwork(state_dim=10, action_dim=5)
        
        # Test forward pass
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_value_network(self):
        """Test value network."""
        network = ValueNetwork(state_dim=10)
        
        # Test forward pass
        state = torch.randn(1, 10)
        output = network(state)
        
        assert output.shape == (1, 1)
        assert isinstance(output, torch.Tensor)
    
    def test_ppo_agent_initialization(self):
        """Test PPO agent initialization."""
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.device.type == "cpu"
        assert agent.gamma == 0.99
    
    def test_ppo_action_selection(self):
        """Test PPO action selection."""
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        state = torch.randn(1, 10)
        
        action, log_prob, value = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestIntegration:
    """Integration tests."""
    
    def test_env_agent_integration(self):
        """Test environment and agent integration."""
        env = JobSchedulingEnv(num_resources=2, max_jobs=3, seed=42)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device="cpu"
        )
        
        # Run one episode
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < 10:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            obs = next_obs
            step_count += 1
        
        assert step_count > 0
        assert len(agent.replay_buffer) > 0


if __name__ == "__main__":
    pytest.main([__file__])
