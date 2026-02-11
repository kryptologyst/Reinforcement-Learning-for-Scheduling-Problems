"""Simple test to verify the project works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from envs.scheduling_env import JobSchedulingEnv
    from algorithms.dqn import DQNAgent
    import torch
    import numpy as np
    
    print("✓ All imports successful!")
    
    # Test environment creation
    env = JobSchedulingEnv(num_resources=2, max_jobs=3, seed=42)
    print("✓ Environment created successfully!")
    
    # Test agent creation
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="cpu"
    )
    print("✓ DQN agent created successfully!")
    
    # Test one step
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    action = agent.select_action(state, training=True)
    next_state, reward, done, truncated, info = env.step(action)
    print("✓ Environment step completed successfully!")
    
    print("\n🎉 Project setup is working correctly!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python scripts/train.py --algorithm dqn --num_episodes 1000")
    print("3. Launch demo: streamlit run demo/app.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
