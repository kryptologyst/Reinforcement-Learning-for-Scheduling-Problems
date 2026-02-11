# Reinforcement Learning for Scheduling Problems

Research-ready implementation of reinforcement learning algorithms for job scheduling optimization. This project demonstrates state-of-the-art RL techniques applied to realistic scheduling problems with comprehensive evaluation, visualization, and interactive demos.

## ⚠️ Safety Notice

**This project is for research and educational purposes only. Do not use for production scheduling systems without proper validation, safety measures, and extensive testing.**

## Features

- **Modern RL Algorithms**: DQN variants (Double DQN, Noisy DQN, Prioritized Experience Replay) and PPO
- **Realistic Environment**: Job shop scheduling with dynamic job arrivals, resource constraints, and deadlines
- **Comprehensive Evaluation**: Statistical analysis, learning curves, and performance metrics
- **Interactive Demo**: Streamlit-based visualization and analysis tool
- **Production-Ready Structure**: Clean code, type hints, configuration management, and testing
- **Reproducible Research**: Deterministic seeding, experiment tracking, and detailed logging

## Project Structure

```
├── src/                          # Source code
│   ├── algorithms/              # RL algorithm implementations
│   │   ├── dqn.py              # DQN variants
│   │   └── ppo.py              # PPO implementation
│   ├── envs/                    # Environment implementations
│   │   └── scheduling_env.py   # Job scheduling environment
│   ├── train/                   # Training scripts
│   │   └── trainer.py          # Main training loop
│   └── utils/                   # Utility modules
│       ├── logging.py          # Logging utilities
│       └── metrics.py          # Metrics tracking
├── configs/                     # Configuration files
│   ├── dqn_config.yaml         # DQN configuration
│   └── ppo_config.yaml         # PPO configuration
├── demo/                        # Interactive demo
│   └── app.py                  # Streamlit demo app
├── scripts/                     # Training and evaluation scripts
├── tests/                       # Unit tests
├── assets/                      # Generated plots and visualizations
├── data/                        # Data storage
└── outputs/                     # Training outputs and checkpoints
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Reinforcement-Learning-for-Scheduling-Problems.git
cd Reinforcement-Learning-for-Scheduling-Problems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train a DQN agent:
```bash
python -m src.train.trainer --algorithm dqn --num_episodes 5000 --experiment_name dqn_test
```

Train a PPO agent:
```bash
python -m src.train.trainer --algorithm ppo --num_episodes 5000 --experiment_name ppo_test
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

Upload a trained model checkpoint and config file to visualize agent performance.

## Environment Description

The `JobSchedulingEnv` simulates a realistic job shop scheduling problem:

- **Jobs**: Arrive dynamically with different priorities, durations, and resource requirements
- **Resources**: Have different capabilities and current workloads
- **Objective**: Maximize scheduling efficiency, meet deadlines, and optimize resource utilization
- **State Space**: Job features + resource features + system features
- **Action Space**: Discrete choice of which job to schedule next (including wait action)

### Key Features

- Dynamic job arrivals with configurable rates
- Resource capability constraints
- Deadline-based urgency calculations
- Realistic reward functions based on scheduling efficiency
- Comprehensive metrics tracking

## Algorithms

### DQN Variants

- **Standard DQN**: Basic deep Q-learning with experience replay
- **Double DQN**: Reduces overestimation bias in Q-values
- **Noisy DQN**: Parameter space noise for exploration
- **Prioritized Experience Replay**: Prioritizes important transitions

### PPO

- **Proximal Policy Optimization**: State-of-the-art policy gradient method
- **Generalized Advantage Estimation**: Efficient advantage estimation
- **Clipped Surrogate Objective**: Stable policy updates
- **Value Function Approximation**: Separate value network

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
# Environment Configuration
env:
  num_resources: 3
  max_jobs: 10
  job_arrival_rate: 0.3
  max_duration: 10.0
  max_priority: 5.0
  time_horizon: 100.0

# Algorithm Configuration
algorithm: "dqn"
learning_rate: 1e-4
gamma: 0.99

# Training Configuration
training:
  num_episodes: 10000
  eval_freq: 100
  eval_episodes: 10
```

## Evaluation Metrics

The project tracks comprehensive performance metrics:

### Learning Metrics
- Episode rewards and returns
- Training loss and convergence
- Sample efficiency
- Learning stability

### Scheduling Metrics
- Resource utilization
- Average wait time
- Deadline satisfaction rate
- Job completion rate
- Queue length statistics

### Statistical Analysis
- Mean ± 95% confidence intervals
- Performance variance
- Robustness across seeds
- Ablation studies

## Usage Examples

### Basic Training

```python
from src.train.trainer import Trainer, TrainingConfig

# Create configuration
config = TrainingConfig(
    algorithm="dqn",
    num_episodes=5000,
    experiment_name="my_experiment"
)

# Train agent
trainer = Trainer(config)
trainer.train()
```

### Custom Environment

```python
from src.envs.scheduling_env import JobSchedulingEnv

# Create custom environment
env = JobSchedulingEnv(
    num_resources=5,
    max_jobs=20,
    job_arrival_rate=0.5,
    max_duration=15.0
)

# Use with any RL algorithm
```

### Evaluation

```python
# Load trained agent
agent = DQNAgent(state_dim, action_dim)
agent.load("checkpoints/best_model.pt")

# Evaluate performance
eval_results = trainer.evaluate(num_episodes=100)
print(f"Average reward: {eval_results['eval_reward_mean']:.2f}")
```

## Advanced Features

### Experiment Tracking

Enable Weights & Biases integration:
```bash
python -m src.train.trainer --use_wandb --experiment_name wandb_test
```

### Hyperparameter Search

Use Optuna for automated hyperparameter optimization:
```python
import optuna

def objective(trial):
    config = TrainingConfig(
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2),
        gamma=trial.suggest_float('gamma', 0.9, 0.999),
        # ... other parameters
    )
    trainer = Trainer(config)
    trainer.train()
    return trainer.best_eval_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Multi-GPU Training

For large-scale experiments:
```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.trainer --device cuda
```

## Development

### Code Quality

The project follows modern Python practices:

- Type hints throughout
- Google-style docstrings
- Black code formatting
- Ruff linting
- Comprehensive testing

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## Results and Benchmarks

### Expected Performance

On the default job scheduling environment:

- **DQN**: ~15-25 average reward after 5000 episodes
- **PPO**: ~20-30 average reward after 5000 episodes
- **Resource Utilization**: 70-85% for well-trained agents
- **Convergence**: Typically within 2000-3000 episodes

### Ablation Studies

The project includes comprehensive ablation studies:

- With/without Double DQN
- Different exploration strategies
- Various network architectures
- Reward function components
- Environment parameter sensitivity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this project in your research, please cite:

```bibtex
@software{rl_scheduling,
  title={Reinforcement Learning for Scheduling Problems},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Reinforcement-Learning-for-Scheduling-Problems}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gymnasium for the RL environment interface
- PyTorch team for the deep learning framework
- Stable Baselines3 for algorithm inspiration
- The RL research community for foundational work

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Enable GPU acceleration or reduce environment complexity
3. **Poor performance**: Check hyperparameters and environment configuration
4. **Import errors**: Ensure all dependencies are installed

### Getting Help

- Check the issues page for common problems
- Review the configuration examples
- Run the test suite to verify installation
- Consult the algorithm-specific documentation

## Future Work

- Multi-agent scheduling scenarios
- Hierarchical RL for complex scheduling
- Transfer learning across scheduling domains
- Real-world scheduling problem integration
- Advanced exploration strategies
- Constraint-aware RL algorithms
# Reinforcement-Learning-for-Scheduling-Problems
