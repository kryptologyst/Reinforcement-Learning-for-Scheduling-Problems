"""Modern Reinforcement Learning Algorithms for Scheduling Problems.

This module implements state-of-the-art RL algorithms adapted for scheduling problems,
including DQN variants, PPO, and specialized scheduling algorithms.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
from dataclasses import dataclass
import math


@dataclass
class Transition:
    """Represents a transition in the replay buffer."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, transition: Transition, priority: float = None) -> None:
        """Add a transition with priority."""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], torch.Tensor, torch.Tensor]:
        """Sample a batch with importance sampling weights."""
        if len(self.buffer) == 0:
            return [], torch.tensor([]), torch.tensor([])
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return transitions, torch.tensor(weights, dtype=torch.float32), torch.tensor(indices)
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for scheduling problems."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1
    ):
        """Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialize noisy linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self) -> None:
        """Reset parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self) -> None:
        """Reset noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class NoisyDQNNetwork(nn.Module):
    """Noisy DQN network for exploration without epsilon-greedy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        """Initialize noisy DQN network."""
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Add noisy layers
        layers.append(NoisyLinear(prev_dim, hidden_dims[-1]))
        layers.append(nn.ReLU())
        layers.append(NoisyLinear(hidden_dims[-1], action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)
    
    def reset_noise(self) -> None:
        """Reset noise in all noisy layers."""
        for module in self.network:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DQNAgent:
    """Deep Q-Network agent with modern improvements."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "auto",
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_noisy: bool = False,
        use_per: bool = False,
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay steps
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            use_double_dqn: Whether to use Double DQN
            use_dueling: Whether to use Dueling DQN
            use_noisy: Whether to use Noisy DQN
            use_per: Whether to use Prioritized Experience Replay
        """
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_noisy = use_noisy
        
        # Epsilon scheduling
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Networks
        if use_noisy:
            self.q_network = NoisyDQNNetwork(state_dim, action_dim).to(self.device)
            self.target_network = NoisyDQNNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
            self.beta_start = 0.4
            self.beta_end = 1.0
            self.beta_decay = 100000
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Training metrics
        self.losses = []
        self.q_values = []
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy exploration."""
        if self.use_noisy:
            # Noisy DQN doesn't need epsilon-greedy
            with torch.no_grad():
                q_values = self.q_network(state.to(self.device))
                return q_values.argmax().item()
        else:
            # Epsilon-greedy exploration
            if training:
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                              math.exp(-self.steps_done / self.epsilon_decay)
                self.steps_done += 1
                
                if random.random() < self.epsilon:
                    return random.randrange(self.action_dim)
            
            with torch.no_grad():
                q_values = self.q_network(state.to(self.device))
                return q_values.argmax().item()
    
    def store_transition(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool) -> None:
        """Store transition in replay buffer."""
        transition = Transition(state, action, reward, next_state, done)
        
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # For PER, we need to estimate priority
            with torch.no_grad():
                q_values = self.q_network(state.to(self.device))
                priority = abs(q_values[action].item()) + 1e-6
            self.replay_buffer.push(transition, priority)
        else:
            self.replay_buffer.push(transition)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            transitions, weights, indices = self.replay_buffer.sample(
                self.batch_size, 
                beta=self._get_beta()
            )
            if not transitions:
                return None
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
            weights = None
            indices = None
        
        # Convert to tensors
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool).to(self.device)
        
        # Compute Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_state_batch).argmax(1)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN
                next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            
            target_q_values = reward_batch.unsqueeze(1) + (self.gamma * next_q_values * ~done_batch.unsqueeze(1))
        
        # Compute loss
        if weights is not None:
            weights = weights.to(self.device)
            loss = (weights.unsqueeze(1) * self.criterion(current_q_values, target_q_values)).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Update priorities for PER
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q_values - target_q_values).squeeze().cpu()
                self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reset noise for Noisy DQN
        if self.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Store metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())
        
        return loss.item()
    
    def _get_beta(self) -> float:
        """Get current beta value for PER."""
        return self.beta_end + (self.beta_start - self.beta_end) * \
               math.exp(-self.steps_done / self.beta_decay)
    
    def save(self, filepath: str) -> None:
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
