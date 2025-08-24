"""
Enhanced Training Script with Live Dashboard Integration
Combines the live progress monitoring with real-time web dashboard visualization
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard'))

from data.data_loader import DataLoader
from environment.trading_env import TradingEnvironment
from environment.continuous_trading_env import ContinuousTradingEnvironment
from agents.dqn_agent import DQNAgent
from agents.sac_agent import SACAgent
from dashboard_integration import DashboardIntegratedMonitor

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_with_dashboard(config_path: str = 'config/config.yaml', agent_type: str = 'sac', episodes_override: int = None):
    """Train agent with live dashboard visualization."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Override episodes if specified
    if episodes_override:
        config['training']['episodes'] = episodes_override
    
    print(f"🚀 Starting {agent_type.upper()} Training with Live Dashboard")
    print(f"📊 Dashboard will be available at: http://localhost:5000")
    print(f"🎯 Episodes: {config['training']['episodes']}")
    print(f"💾 Symbol: {config['data']['symbols'][0]}")
    
    # Load data
    data_loader = DataLoader(config)
    train_data = data_loader.load_data('train')
    
    if train_data is None or train_data.empty:
        raise ValueError("No training data loaded")
    
    print(f"📈 Loaded {len(train_data)} training samples")
    
    # Create environment based on agent type
    if agent_type.lower() == 'dqn':
        env = TradingEnvironment(train_data, config)
        print("🎮 Using discrete action environment for DQN")
    else:  # SAC
        env = ContinuousTradingEnvironment(train_data, config)
        print("🎮 Using continuous action environment for SAC")
    
    # Create agent
    state_size = env.observation_space.shape[0]
    
    if agent_type.lower() == 'dqn':
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size, config)
        print(f"🤖 Created DQN agent (state: {state_size}, actions: {action_size})")
    else:  # SAC
        action_size = env.action_space.shape[0]
        agent = SACAgent(state_size, action_size, config)
        print(f"🤖 Created SAC agent (state: {state_size}, action_dim: {action_size})")
    
    # Initialize dashboard monitor
    monitor = DashboardIntegratedMonitor(
        total_episodes=config['training']['episodes'],
        agent_type=agent_type,
        enable_dashboard=True
    )
    
    print("📊 Dashboard integration enabled")
    print("🔗 Starting dashboard server in background...")
    
    # Start dashboard server in background
    import subprocess
    import threading
    
    def start_dashboard():
        try:
            dashboard_process = subprocess.Popen([
                sys.executable, 
                os.path.join('dashboard', 'app.py')
            ], cwd=os.path.dirname(__file__))
            return dashboard_process
        except Exception as e:
            print(f"⚠️  Could not start dashboard: {e}")
            return None
    
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Give dashboard time to start
    time.sleep(3)
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    episode_portfolio_values = []
    episode_times = []
    
    symbol = config['data']['symbols'][0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n🎯 Starting Training Loop...")
    print("=" * 80)
    
    try:
        for episode in range(config['training']['episodes']):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            monitor.start_episode(episode)
            episode_reward = 0
            episode_start_time = time.time()
            step_count = 0
            
            # Track episode data for dashboard
            episode_portfolio_history = []
            episode_price_history = []
            
            while True:
                # Agent action
                if agent_type.lower() == 'dqn':
                    action = agent.act(state)
                    epsilon = agent.epsilon
                    alpha = None
                else:  # SAC
                    action = agent.select_action(state)
                    epsilon = None
                    alpha = agent.alpha
                
                # Environment step
                step_result = env.step(action)
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                    truncated = False
                else:
                    next_state, reward, done, truncated, info = step_result
                
                episode_reward += reward
                step_count += 1
                
                # Store experience and train
                loss = None
                if agent_type.lower() == 'dqn':
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > config['training']['batch_size']:
                        loss = agent.replay(config['training']['batch_size'])
                else:  # SAC
                    agent.store_transition(state, action, reward, next_state, done)
                    if len(agent.replay_buffer.buffer) > config['training']['batch_size']:
                        losses = agent.update_parameters()
                        loss = losses.get('critic1_loss', 0) if isinstance(losses, dict) else losses
                
                # Get current portfolio and price info
                current_portfolio = info.get('portfolio_value', env.portfolio_value)
                current_price = info.get('current_price', env.current_price)
                
                # Track for dashboard
                episode_portfolio_history.append(current_portfolio)
                episode_price_history.append(current_price)
                
                # Calculate positions
                positions = {
                    'cash': info.get('cash', env.cash),
                    'shares': info.get('shares', env.shares),
                    'total_value': current_portfolio
                }
                
                # Update monitor with dashboard integration
                monitor.update_step(
                    action=action,
                    reward=reward,
                    portfolio_value=current_portfolio,
                    current_price=current_price,
                    epsilon_or_alpha=epsilon if epsilon is not None else alpha,
                    loss=loss,
                    positions=positions
                )
                
                # Print live progress every 50 steps
                if step_count % 50 == 0:
                    monitor.print_live_progress()
                
                state = next_state
                
                if done or truncated:
                    break
            
            # Episode finished
            episode_time = time.time() - episode_start_time
            final_portfolio = env.portfolio_value
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            episode_portfolio_values.append(final_portfolio)
            episode_times.append(episode_time)
            
            # Finish episode for dashboard
            monitor.finish_episode()
            
            # Print episode summary
            if episode % 10 == 0 or episode == config['training']['episodes'] - 1:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_portfolio = np.mean(episode_portfolio_values[-10:])
                print(f"\n📊 Episode {episode:4d} | Reward: {episode_reward:8.2f} | Portfolio: ${final_portfolio:8.0f}")
                print(f"    Avg Reward (10): {avg_reward:8.2f} | Avg Portfolio (10): ${avg_portfolio:8.0f}")
                print(f"    Steps: {step_count:4d} | Time: {episode_time:.2f}s")
                
                if agent_type.lower() == 'dqn':
                    print(f"    Epsilon: {agent.epsilon:.3f}")
                else:
                    print(f"    Alpha: {agent.alpha:.3f}")
        
        print("\n🎉 Training completed!")
        
        # Save model
        model_path = f'models/{agent_type}_agent_{symbol}_{timestamp}.pth'
        if agent_type.lower() == 'dqn':
            agent.save(model_path)
        else:
            agent.save_models(model_path.replace('.pth', ''))
        
        print(f"💾 Model saved: {model_path}")
        
        # Generate final plots
        create_training_plots(episode_rewards, episode_portfolio_values, 
                            episode_lengths, episode_times, symbol, timestamp, agent_type)
        
        # Final results
        total_return = ((episode_portfolio_values[-1] - episode_portfolio_values[0]) / episode_portfolio_values[0]) * 100
        avg_episode_reward = np.mean(episode_rewards)
        
        results = {
            'agent_type': agent_type,
            'total_episodes': len(episode_rewards),
            'final_portfolio': episode_portfolio_values[-1],
            'total_return': total_return,
            'avg_episode_reward': avg_episode_reward,
            'model_path': model_path
        }
        
        print(f"\n📈 Final Results:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Final Portfolio: ${episode_portfolio_values[-1]:,.2f}")
        print(f"   Average Episode Reward: {avg_episode_reward:.2f}")
        print(f"\n🌐 Dashboard remains active at: http://localhost:5000")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise

def create_training_plots(episode_rewards, episode_portfolio_values, episode_lengths, 
                         episode_times, symbol, timestamp, agent_type):
    """Create comprehensive training plots."""
    
    os.makedirs('logs/plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{agent_type.upper()} Training Progress - {symbol}', fontsize=16)
    
    episodes = range(len(episode_rewards))
    
    # Episode rewards
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.7, color='blue')
    if len(episode_rewards) > 10:
        window = min(10, len(episode_rewards) // 4)
        ma_rewards = pd.Series(episode_rewards).rolling(window).mean()
        axes[0, 0].plot(episodes, ma_rewards, color='red', linewidth=2, label=f'{window}-episode MA')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio values
    axes[0, 1].plot(episodes, episode_portfolio_values, alpha=0.7, color='green')
    if len(episode_portfolio_values) > 10:
        window = min(10, len(episode_portfolio_values) // 4)
        ma_portfolio = pd.Series(episode_portfolio_values).rolling(window).mean()
        axes[0, 1].plot(episodes, ma_portfolio, color='red', linewidth=2, label=f'{window}-episode MA')
        axes[0, 1].legend()
    axes[0, 1].set_title('Portfolio Value')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(episodes, episode_lengths, alpha=0.7, color='orange')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode times
    axes[1, 1].plot(episodes, episode_times, alpha=0.7, color='purple')
    if len(episode_times) > 10:
        window = min(10, len(episode_times) // 4)
        ma_times = pd.Series(episode_times).rolling(window).mean()
        axes[1, 1].plot(episodes, ma_times, color='red', linewidth=2, label=f'{window}-episode MA')
        axes[1, 1].legend()
    axes[1, 1].set_title('Episode Duration')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'logs/plots/dashboard_training_{agent_type}_{symbol}_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved: {plot_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL Trading Agent with Live Dashboard')
    parser.add_argument('--agent', '-a', type=str, choices=['dqn', 'sac'], default='sac',
                        help='Agent type to use (default: sac)')
    parser.add_argument('--config', '-c', type=str, default='config/config.yaml',
                        help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='Number of episodes to train (overrides config)')
    return parser.parse_args()

if __name__ == "__main__":
    import torch
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"🎯 Selected Agent: {args.agent.upper()}")
    print(f"⚙️  Config File: {args.config}")
    if args.episodes:
        print(f"📊 Episodes: {args.episodes}")
    
    try:
        results = train_with_dashboard(
            config_path=args.config, 
            agent_type=args.agent,
            episodes_override=args.episodes
        )
        
        if results:
            print(f"\n✅ Training completed successfully!")
            print(f"🎯 Agent: {results['agent_type'].upper()}")
            print(f"📊 Episodes: {results['total_episodes']}")
            print(f"💰 Final Portfolio: ${results['final_portfolio']:,.2f}")
            print(f"📈 Total Return: {results['total_return']:.2f}%")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
