"""Proximal Policy Optimization (PPO) for Scheduling Problems.

This module implements PPO, a state-of-the-art policy gradient algorithm
that works well for both discrete and continuous action spaces.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import math


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.LayerNorm(hidden_dim)  # Layer normalization for stability
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.state_dim = state_dim
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class PPOBuffer:
    """Buffer for storing PPO experiences."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        """Initialize PPO buffer.
        
        Args:
            buffer_size: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize buffers
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def store(
        self, 
        state: torch.Tensor, 
        action: int, 
        reward: float, 
        value: float, 
        log_prob: float
    ) -> None:
        """Store experience in buffer."""
        assert self.ptr < self.buffer_size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all stored experiences."""
        assert self.size == self.buffer_size
        
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns
        }
    
    def compute_advantages_and_returns(
        self, 
        next_value: float, 
        gamma: float = 0.99, 
        lam: float = 0.95
    ) -> None:
        """Compute advantages and returns using GAE."""
        rewards = torch.cat([self.rewards, torch.tensor([next_value], device=self.device)])
        values = torch.cat([self.values, torch.tensor([next_value], device=self.device)])
        
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        buffer_size: int = 2048,
        batch_size: int = 64,
        device: str = "auto",
    ):
        """Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            lam: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            buffer_size: Buffer size for collecting experiences
            batch_size: Training batch size
            device: Device to use
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
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Networks
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_network = ValueNetwork(state_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Buffer
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.kl_divergences = []
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy.
        
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: State value estimate
        """
        state = state.to(self.device)
        
        with torch.no_grad():
            # Get action probabilities
            logits = self.policy_network(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get state value
            value = self.value_network(state)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_value(self, state: torch.Tensor) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            value = self.value_network(state.to(self.device))
        return value.item()
    
    def store_experience(
        self, 
        state: torch.Tensor, 
        action: int, 
        reward: float, 
        value: float, 
        log_prob: float
    ) -> None:
        """Store experience in buffer."""
        self.buffer.store(state, action, reward, value, log_prob)
    
    def update(self, next_state: torch.Tensor) -> Dict[str, float]:
        """Update policy and value networks."""
        if self.buffer.size < self.buffer.buffer_size:
            return {}
        
        # Compute advantages and returns
        next_value = self.get_value(next_state)
        self.buffer.compute_advantages_and_returns(next_value, self.gamma, self.lam)
        
        # Get batch data
        batch = self.buffer.get()
        
        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divergences = []
        
        # PPO updates
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(self.buffer.buffer_size, device=self.device)
            
            for start_idx in range(0, self.buffer.buffer_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.buffer.buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                states = batch['states'][batch_indices]
                actions = batch['actions'][batch_indices]
                old_log_probs = batch['log_probs'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns = batch['returns'][batch_indices]
                
                # Forward pass
                logits = self.policy_network(states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                values = self.value_network(states).squeeze()
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, returns)
                
                # Compute KL divergence
                kl_div = (old_log_probs - new_log_probs).mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_loss_coef * value_loss - 
                            self.entropy_coef * entropy)
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kl_divergences.append(kl_div.item())
        
        # Clear buffer
        self.buffer.clear()
        
        # Store metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divergences)
        }
        
        self.policy_losses.append(metrics['policy_loss'])
        self.value_losses.append(metrics['value_loss'])
        self.entropies.append(metrics['entropy'])
        self.kl_divergences.append(metrics['kl_divergence'])
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
