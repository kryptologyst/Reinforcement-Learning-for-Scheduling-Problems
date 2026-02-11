"""Scheduling Environment for Reinforcement Learning.

This module implements a realistic job scheduling environment compatible with Gymnasium.
The environment simulates a job shop scheduling problem where tasks must be assigned
to resources with different capabilities and constraints.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from dataclasses import dataclass


@dataclass
class Job:
    """Represents a job/task in the scheduling system."""
    id: int
    duration: float
    priority: float
    resource_requirements: List[int]  # Which resources can handle this job
    deadline: Optional[float] = None
    arrival_time: float = 0.0


@dataclass
class Resource:
    """Represents a resource/machine in the scheduling system."""
    id: int
    capability: int  # Resource capability level
    current_job: Optional[int] = None
    completion_time: float = 0.0
    utilization: float = 0.0


class JobSchedulingEnv(gym.Env):
    """Job Shop Scheduling Environment for Reinforcement Learning.
    
    This environment simulates a realistic job scheduling problem where:
    - Jobs arrive dynamically with different priorities, durations, and requirements
    - Resources have different capabilities and current workloads
    - The agent must decide which job to schedule next
    - Rewards are based on scheduling efficiency, deadline adherence, and resource utilization
    
    State Space:
        - Job features: [duration, priority, deadline_urgency, resource_compatibility]
        - Resource features: [capability, current_load, utilization]
        - System features: [time, queue_length, avg_wait_time]
    
    Action Space:
        - Discrete: Choose which job to schedule next (including "wait" action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_resources: int = 3,
        max_jobs: int = 10,
        job_arrival_rate: float = 0.3,
        max_duration: float = 10.0,
        max_priority: float = 5.0,
        time_horizon: float = 100.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the job scheduling environment.
        
        Args:
            num_resources: Number of available resources/machines
            max_jobs: Maximum number of jobs in the system
            job_arrival_rate: Probability of new job arrival per step
            max_duration: Maximum job duration
            max_priority: Maximum job priority
            time_horizon: Maximum simulation time
            render_mode: Rendering mode for visualization
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_resources = num_resources
        self.max_jobs = max_jobs
        self.job_arrival_rate = job_arrival_rate
        self.max_duration = max_duration
        self.max_priority = max_priority
        self.time_horizon = time_horizon
        self.render_mode = render_mode
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Action space: choose job to schedule (including wait action)
        self.action_space = spaces.Discrete(max_jobs + 1)  # +1 for "wait" action
        
        # State space: job features + resource features + system features
        job_features = 4  # duration, priority, deadline_urgency, resource_compatibility
        resource_features = 3  # capability, current_load, utilization
        system_features = 3  # time, queue_length, avg_wait_time
        
        state_dim = (max_jobs * job_features + 
                    num_resources * resource_features + 
                    system_features)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Environment state
        self.jobs: List[Job] = []
        self.resources: List[Resource] = []
        self.current_time = 0.0
        self.total_reward = 0.0
        self.completed_jobs = 0
        self.missed_deadlines = 0
        
        # Performance metrics
        self.metrics = {
            'total_completion_time': 0.0,
            'avg_wait_time': 0.0,
            'resource_utilization': 0.0,
            'deadline_satisfaction': 1.0,
        }
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Initialize resources
        self.resources = [
            Resource(
                id=i,
                capability=self.np_random.uniform(0.5, 1.5),
                current_job=None,
                completion_time=0.0,
                utilization=0.0
            )
            for i in range(self.num_resources)
        ]
        
        # Initialize jobs queue
        self.jobs = []
        self._generate_initial_jobs()
        
        # Reset state variables
        self.current_time = 0.0
        self.total_reward = 0.0
        self.completed_jobs = 0
        self.missed_deadlines = 0
        
        # Reset metrics
        self.metrics = {
            'total_completion_time': 0.0,
            'avg_wait_time': 0.0,
            'resource_utilization': 0.0,
            'deadline_satisfaction': 1.0,
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Update current time
        self.current_time += 1.0
        
        # Process resource completions
        self._process_resource_completions()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Generate new jobs
        if self.np_random.random() < self.job_arrival_rate:
            self._generate_new_job()
        
        # Update metrics
        self._update_metrics()
        
        # Check termination conditions
        terminated = self.current_time >= self.time_horizon
        truncated = len(self.jobs) == 0 and all(r.current_job is None for r in self.resources)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_initial_jobs(self) -> None:
        """Generate initial jobs in the system."""
        num_initial_jobs = self.np_random.randint(1, min(5, self.max_jobs))
        for i in range(num_initial_jobs):
            self._generate_new_job()
    
    def _generate_new_job(self) -> None:
        """Generate a new job and add it to the queue."""
        if len(self.jobs) >= self.max_jobs:
            return
            
        job_id = len(self.jobs)
        duration = self.np_random.uniform(1.0, self.max_duration)
        priority = self.np_random.uniform(1.0, self.max_priority)
        
        # Random deadline (some jobs have deadlines, others don't)
        deadline = None
        if self.np_random.random() < 0.7:  # 70% of jobs have deadlines
            deadline = self.current_time + self.np_random.uniform(duration * 1.5, duration * 3.0)
        
        # Resource requirements (which resources can handle this job)
        num_compatible_resources = self.np_random.randint(1, self.num_resources + 1)
        resource_requirements = self.np_random.choice(
            self.num_resources, 
            size=num_compatible_resources, 
            replace=False
        ).tolist()
        
        job = Job(
            id=job_id,
            duration=duration,
            priority=priority,
            resource_requirements=resource_requirements,
            deadline=deadline,
            arrival_time=self.current_time
        )
        
        self.jobs.append(job)
    
    def _execute_action(self, action: int) -> float:
        """Execute the scheduling action and return reward."""
        if action == 0:  # Wait action
            return -0.1  # Small penalty for waiting
        
        job_idx = action - 1
        if job_idx >= len(self.jobs):
            return -0.5  # Penalty for invalid action
        
        job = self.jobs[job_idx]
        
        # Find best available resource for this job
        best_resource = self._find_best_resource(job)
        if best_resource is None:
            return -0.3  # Penalty for no available resource
        
        # Schedule the job
        best_resource.current_job = job.id
        best_resource.completion_time = self.current_time + job.duration
        
        # Remove job from queue
        self.jobs.pop(job_idx)
        
        # Calculate reward based on scheduling efficiency
        reward = self._calculate_scheduling_reward(job, best_resource)
        
        return reward
    
    def _find_best_resource(self, job: Job) -> Optional[Resource]:
        """Find the best available resource for a job."""
        available_resources = [
            r for r in self.resources 
            if r.current_job is None and r.id in job.resource_requirements
        ]
        
        if not available_resources:
            return None
        
        # Choose resource with highest capability
        return max(available_resources, key=lambda r: r.capability)
    
    def _calculate_scheduling_reward(self, job: Job, resource: Resource) -> float:
        """Calculate reward for scheduling a job on a resource."""
        reward = 0.0
        
        # Base reward for successful scheduling
        reward += 1.0
        
        # Priority bonus
        reward += job.priority * 0.2
        
        # Resource efficiency bonus
        efficiency = resource.capability / job.duration
        reward += efficiency * 0.5
        
        # Deadline urgency bonus
        if job.deadline is not None:
            time_to_deadline = job.deadline - self.current_time
            urgency = max(0, (job.duration * 2 - time_to_deadline) / job.duration)
            reward += urgency * 0.3
        
        # Wait time penalty (jobs that waited longer get higher priority)
        wait_time = self.current_time - job.arrival_time
        reward += min(wait_time * 0.1, 1.0)  # Cap the bonus
        
        return reward
    
    def _process_resource_completions(self) -> None:
        """Process completed jobs on resources."""
        for resource in self.resources:
            if (resource.current_job is not None and 
                self.current_time >= resource.completion_time):
                
                # Job completed
                self.completed_jobs += 1
                
                # Check if deadline was missed
                # Note: We don't have job details here, so we'll track this differently
                
                # Free up resource
                resource.current_job = None
                resource.completion_time = 0.0
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        # Resource utilization
        busy_resources = sum(1 for r in self.resources if r.current_job is not None)
        self.metrics['resource_utilization'] = busy_resources / self.num_resources
        
        # Average wait time
        if self.jobs:
            wait_times = [self.current_time - job.arrival_time for job in self.jobs]
            self.metrics['avg_wait_time'] = np.mean(wait_times)
        else:
            self.metrics['avg_wait_time'] = 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Job features (pad with zeros if fewer than max_jobs)
        for i in range(self.max_jobs):
            if i < len(self.jobs):
                job = self.jobs[i]
                obs.extend([
                    job.duration / self.max_duration,  # Normalized duration
                    job.priority / self.max_priority,  # Normalized priority
                    self._get_deadline_urgency(job),   # Deadline urgency
                    self._get_resource_compatibility(job)  # Resource compatibility
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # Padding
        
        # Resource features
        for resource in self.resources:
            obs.extend([
                resource.capability,
                resource.completion_time / self.time_horizon,  # Normalized completion time
                resource.utilization
            ])
        
        # System features
        obs.extend([
            self.current_time / self.time_horizon,  # Normalized time
            len(self.jobs) / self.max_jobs,        # Normalized queue length
            self.metrics['avg_wait_time'] / self.max_duration  # Normalized wait time
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_deadline_urgency(self, job: Job) -> float:
        """Calculate deadline urgency for a job."""
        if job.deadline is None:
            return 0.0
        
        time_to_deadline = job.deadline - self.current_time
        if time_to_deadline <= 0:
            return 1.0  # Overdue
        
        urgency = max(0, (job.duration - time_to_deadline) / job.duration)
        return min(urgency, 1.0)
    
    def _get_resource_compatibility(self, job: Job) -> float:
        """Calculate resource compatibility score for a job."""
        compatible_resources = [
            r for r in self.resources 
            if r.id in job.resource_requirements and r.current_job is None
        ]
        return len(compatible_resources) / self.num_resources
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'current_time': self.current_time,
            'num_jobs': len(self.jobs),
            'completed_jobs': self.completed_jobs,
            'missed_deadlines': self.missed_deadlines,
            'metrics': self.metrics.copy(),
            'resource_states': [
                {
                    'id': r.id,
                    'capability': r.capability,
                    'current_job': r.current_job,
                    'completion_time': r.completion_time,
                    'utilization': r.utilization
                }
                for r in self.resources
            ]
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Time: {self.current_time:.1f}")
            print(f"Jobs in queue: {len(self.jobs)}")
            print(f"Completed jobs: {self.completed_jobs}")
            print(f"Resource utilization: {self.metrics['resource_utilization']:.2f}")
            print(f"Average wait time: {self.metrics['avg_wait_time']:.2f}")
            print("-" * 40)
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
