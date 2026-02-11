"""Interactive Streamlit demo for RL Scheduling Agent.

This demo provides a user-friendly interface to visualize and interact with
trained RL scheduling agents.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import sys
import json
import yaml
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from envs.scheduling_env import JobSchedulingEnv
from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent


def load_agent(model_path: str, config_path: str, algorithm: str):
    """Load trained agent from checkpoint."""
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
            device="cpu",  # Use CPU for demo
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
            device="cpu",  # Use CPU for demo
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load trained weights
    agent.load(model_path)
    
    return agent, env


def run_episode(agent, env, max_steps: int = 100):
    """Run a single episode and collect data."""
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    episode_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'job_counts': [],
        'resource_utilization': [],
        'avg_wait_time': [],
        'completed_jobs': [],
        'step_info': []
    }
    
    total_reward = 0
    step = 0
    
    while step < max_steps:
        # Select action
        if hasattr(agent, 'select_action'):
            if len(agent.select_action.__code__.co_varnames) > 2:  # PPO
                action, _, _ = agent.select_action(state, training=False)
            else:  # DQN
                action = agent.select_action(state, training=False)
        else:
            action = 0  # Default action
        
        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # Store data
        episode_data['states'].append(state.numpy())
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['job_counts'].append(len(env.jobs))
        episode_data['resource_utilization'].append(info['metrics']['resource_utilization'])
        episode_data['avg_wait_time'].append(info['metrics']['avg_wait_time'])
        episode_data['completed_jobs'].append(info['completed_jobs'])
        episode_data['step_info'].append(info)
        
        total_reward += reward
        step += 1
        
        if done or truncated:
            break
        
        state = next_state
    
    return episode_data, total_reward


def create_performance_plots(episode_data: Dict[str, List]):
    """Create performance visualization plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rewards Over Time', 'Job Queue Length', 
                       'Resource Utilization', 'Average Wait Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    steps = list(range(len(episode_data['rewards'])))
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=steps, y=episode_data['rewards'], 
                  mode='lines', name='Reward', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Job queue length
    fig.add_trace(
        go.Scatter(x=steps, y=episode_data['job_counts'], 
                  mode='lines', name='Jobs in Queue', line=dict(color='green')),
        row=1, col=2
    )
    
    # Resource utilization
    fig.add_trace(
        go.Scatter(x=steps, y=episode_data['resource_utilization'], 
                  mode='lines', name='Resource Utilization', line=dict(color='red')),
        row=2, col=1
    )
    
    # Average wait time
    fig.add_trace(
        go.Scatter(x=steps, y=episode_data['avg_wait_time'], 
                  mode='lines', name='Avg Wait Time', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Episode Performance Metrics")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Value")
    
    return fig


def create_action_distribution_plot(episode_data: Dict[str, List]):
    """Create action distribution visualization."""
    actions = episode_data['actions']
    action_counts = pd.Series(actions).value_counts().sort_index()
    
    fig = px.bar(
        x=action_counts.index, 
        y=action_counts.values,
        title="Action Distribution",
        labels={'x': 'Action', 'y': 'Count'}
    )
    
    return fig


def create_state_visualization(episode_data: Dict[str, List], step: int):
    """Create state visualization for a specific step."""
    if step >= len(episode_data['states']):
        return None
    
    state = episode_data['states'][step]
    step_info = episode_data['step_info'][step]
    
    # Parse state components
    num_jobs = len(step_info.get('resource_states', []))
    num_resources = len(step_info.get('resource_states', []))
    
    # Create state visualization
    fig = go.Figure()
    
    # Add resource utilization bars
    resource_states = step_info.get('resource_states', [])
    resource_ids = [f"Resource {r['id']}" for r in resource_states]
    utilizations = [r['utilization'] for r in resource_states]
    
    fig.add_trace(go.Bar(
        x=resource_ids,
        y=utilizations,
        name='Resource Utilization',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"System State at Step {step}",
        xaxis_title="Resources",
        yaxis_title="Utilization",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Scheduling Agent Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning Scheduling Agent Demo")
    st.markdown("""
    This demo allows you to interact with trained RL agents for job scheduling problems.
    Upload a trained model and visualize its performance in real-time.
    """)
    
    # Sidebar for model upload
    st.sidebar.header("Model Configuration")
    
    # Model upload
    uploaded_model = st.sidebar.file_uploader(
        "Upload Model Checkpoint (.pt file)",
        type=['pt'],
        help="Upload a trained model checkpoint"
    )
    
    uploaded_config = st.sidebar.file_uploader(
        "Upload Config File (.yaml)",
        type=['yaml', 'yml'],
        help="Upload the corresponding config file"
    )
    
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["dqn", "ppo"],
        help="Select the algorithm used to train the model"
    )
    
    # Demo parameters
    st.sidebar.header("Demo Parameters")
    max_steps = st.sidebar.slider("Max Steps per Episode", 10, 200, 100)
    num_episodes = st.sidebar.slider("Number of Episodes", 1, 10, 3)
    
    # Main content
    if uploaded_model and uploaded_config:
        try:
            # Save uploaded files temporarily
            model_path = f"temp_model_{algorithm}.pt"
            config_path = f"temp_config_{algorithm}.yaml"
            
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            with open(config_path, "wb") as f:
                f.write(uploaded_config.getbuffer())
            
            # Load agent
            with st.spinner("Loading agent..."):
                agent, env = load_agent(model_path, config_path, algorithm)
            
            st.success(f"Successfully loaded {algorithm.upper()} agent!")
            
            # Run episodes
            if st.button("Run Episodes", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_episode_data = []
                all_rewards = []
                
                for episode in range(num_episodes):
                    status_text.text(f"Running episode {episode + 1}/{num_episodes}...")
                    
                    episode_data, total_reward = run_episode(agent, env, max_steps)
                    all_episode_data.append(episode_data)
                    all_rewards.append(total_reward)
                    
                    progress_bar.progress((episode + 1) / num_episodes)
                
                status_text.text("Episodes completed!")
                
                # Display results
                st.header("Episode Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Reward", f"{np.mean(all_rewards):.2f}")
                with col2:
                    st.metric("Best Reward", f"{np.max(all_rewards):.2f}")
                with col3:
                    st.metric("Worst Reward", f"{np.min(all_rewards):.2f}")
                with col4:
                    st.metric("Std Reward", f"{np.std(all_rewards):.2f}")
                
                # Episode selection
                selected_episode = st.selectbox(
                    "Select Episode to Visualize",
                    range(num_episodes),
                    format_func=lambda x: f"Episode {x + 1} (Reward: {all_rewards[x]:.2f})"
                )
                
                episode_data = all_episode_data[selected_episode]
                
                # Performance plots
                st.header("Performance Visualization")
                perf_fig = create_performance_plots(episode_data)
                st.plotly_chart(perf_fig, use_container_width=True)
                
                # Action distribution
                st.header("Action Distribution")
                action_fig = create_action_distribution_plot(episode_data)
                st.plotly_chart(action_fig, use_container_width=True)
                
                # State visualization
                st.header("State Visualization")
                step_slider = st.slider(
                    "Select Step",
                    0, 
                    len(episode_data['states']) - 1,
                    0
                )
                
                state_fig = create_state_visualization(episode_data, step_slider)
                if state_fig:
                    st.plotly_chart(state_fig, use_container_width=True)
                
                # Raw data
                if st.checkbox("Show Raw Data"):
                    st.header("Raw Episode Data")
                    
                    # Create DataFrame
                    df_data = {
                        'Step': range(len(episode_data['rewards'])),
                        'Action': episode_data['actions'],
                        'Reward': episode_data['rewards'],
                        'Jobs in Queue': episode_data['job_counts'],
                        'Resource Utilization': episode_data['resource_utilization'],
                        'Avg Wait Time': episode_data['avg_wait_time'],
                        'Completed Jobs': episode_data['completed_jobs']
                    }
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df)
            
            # Clean up temporary files
            Path(model_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.exception(e)
    
    else:
        st.info("""
        👆 Please upload a model checkpoint (.pt file) and config file (.yaml) to get started.
        
        **How to use this demo:**
        1. Train a model using the training script
        2. Upload the saved model checkpoint and config file
        3. Select the algorithm used (DQN or PPO)
        4. Configure demo parameters
        5. Click "Run Episodes" to see the agent in action!
        
        **Features:**
        - Real-time performance visualization
        - Action distribution analysis
        - State visualization at each step
        - Raw data export
        - Multiple episode comparison
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Safety Notice:** This demo is for research and educational purposes only. 
    Do not use for production scheduling systems without proper validation and safety measures.
    """)


if __name__ == "__main__":
    main()
