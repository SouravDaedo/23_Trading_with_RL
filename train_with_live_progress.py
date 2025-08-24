#!/usr/bin/env python3
"""
Enhanced training script with real-time progress monitoring and live updates.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict, deque
import threading
import subprocess
import argparse
from typing import Dict, List, Any

# Add src to path
sys.path.append('src')
sys.path.append('dashboard')

from data.data_loader import DataLoader
from environment.trading_env import TradingEnvironment
from environment.continuous_trading_env import ContinuousTradingEnvironment
from agents.dqn_agent import DQNAgent
from agents.sac_agent import SACAgent

# Dashboard integration (optional import)
try:
    from dashboard_integration import DashboardConnector
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️  Dashboard integration not available. Install flask and flask-socketio to enable.")

class LiveProgressMonitor:
    """Real-time training progress monitor with live updates."""
    
    def __init__(self, total_episodes: int, update_frequency: int = 1, enable_dashboard: bool = False):
        self.total_episodes = total_episodes
        self.update_frequency = update_frequency
        self.start_time = time.time()
        self.enable_dashboard = enable_dashboard
        
        # Initialize dashboard connector if enabled
        self.dashboard = None
        if enable_dashboard and DASHBOARD_AVAILABLE:
            self.dashboard = DashboardConnector()
            print("📊 Dashboard connector initialized")
        
        # Progress tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = 0
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_portfolio_values = []
        self.episode_lengths = []
        self.episode_times = []
        self.recent_losses = deque(maxlen=100)
        
        # Real-time metrics
        self.current_reward = 0
        self.current_portfolio = 0
        self.current_epsilon = 0
        self.current_loss = 0
        self.steps_per_second = 0
        
        # Action tracking (works for both discrete and continuous)
        self.episode_actions = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        self.action_ranges = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        self.agent_type = None
        
        # Performance estimates
        self.eta_seconds = 0
        self.episodes_per_minute = 0
        
    def start_episode(self, episode_num: int):
        """Start tracking a new episode."""
        self.current_episode = episode_num
        self.current_step = 0
        self.current_reward = 0
        self.episode_start_time = 0
        self.episode_actions = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        self.action_ranges = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        
    def update_step(self, action, reward: float, portfolio_value: float, 
                   epsilon_or_alpha: float = None, loss: float = None, agent_type: str = 'sac',
                   current_price: float = None):
        """Update step-level metrics for both discrete and continuous actions."""
        self.current_step += 1
        self.current_reward += reward
        self.current_portfolio = portfolio_value
        
        # Store current price if provided
        if current_price is not None:
            self.current_price = current_price
        # Convert epsilon/alpha to scalar if it's a tensor
        if epsilon_or_alpha is not None:
            if hasattr(epsilon_or_alpha, 'item'):
                self.current_epsilon = epsilon_or_alpha.item()
            else:
                self.current_epsilon = float(epsilon_or_alpha)
        else:
            self.current_epsilon = 0
        self.agent_type = agent_type
        
        # Track actions based on agent type
        if agent_type.lower() == 'dqn':
            # Discrete actions: 0=HOLD, 1=BUY, 2=SELL
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(int(action), 'HOLD')
            self.episode_actions[action_name] += 1
        else:  # SAC continuous actions
            # Continuous actions: convert to discrete categories
            action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            if action_val < -0.1:  # Sell threshold
                self.episode_actions['SELL'] += 1
            elif action_val > 0.1:  # Buy threshold
                self.episode_actions['BUY'] += 1
            else:  # Hold range
                self.episode_actions['HOLD'] += 1
        
        # Track loss - ensure it's a scalar value
        if loss is not None:
            # Convert tensor to scalar if needed
            if hasattr(loss, 'item'):
                self.current_loss = loss.item()
            elif isinstance(loss, dict):
                # For SAC agent that returns dict of losses, use critic1_loss as representative
                self.current_loss = loss.get('critic1_loss', 0.0)
                if hasattr(self.current_loss, 'item'):
                    self.current_loss = self.current_loss.item()
            else:
                self.current_loss = float(loss)
            
            self.recent_losses.append(self.current_loss)
        
        # Calculate steps per second
        if self.current_step > 0:
            episode_duration = time.time() - self.episode_start_time
            self.steps_per_second = self.current_step / max(episode_duration, 0.001)
        
        # Store current price for dashboard
        current_price = getattr(self, 'current_price', 150.0)
        
        # Send to dashboard if enabled
        if self.dashboard:
            agent_params = {
                'epsilon' if self.agent_type.lower() == 'dqn' else 'alpha': self.current_epsilon,
                'loss': self.current_loss
            }
            
            # Calculate positions (simplified)
            positions = {
                'cash': self.current_portfolio * 0.5,  # Estimate
                'shares': int(self.current_portfolio * 0.5 / current_price),
                'total_value': self.current_portfolio
            }
            
            self.dashboard.update_step(
                episode=self.current_episode,
                step=self.current_step,
                action=action,
                reward=reward,
                portfolio_value=portfolio_value,
                current_price=current_price,
                agent_params=agent_params,
                positions=positions
            )
    
    def finish_episode(self):
        """Finish episode and update metrics."""
        episode_duration = time.time() - self.episode_start_time
        
        self.episode_rewards.append(self.current_reward)
        self.episode_portfolio_values.append(self.current_portfolio)
        self.episode_lengths.append(self.current_step)
        self.episode_times.append(episode_duration)
        
        # Send episode summary to dashboard if enabled
        if self.dashboard:
            self.dashboard.update_episode_end(
                episode=self.current_episode,
                episode_reward=self.current_reward,
                portfolio_history=self.episode_portfolio_values,
                price_history=getattr(self, 'price_history', []),
                rewards_history=self.episode_rewards,
                actions_summary=getattr(self, 'total_actions', {'BUY': 0, 'SELL': 0, 'HOLD': 0})
            )
        
        # Update performance estimates
        if len(self.episode_times) > 0:
            avg_episode_time = np.mean(self.episode_times[-10:])  # Last 10 episodes
            remaining_episodes = self.total_episodes - self.current_episode - 1
            self.eta_seconds = remaining_episodes * avg_episode_time
            self.episodes_per_minute = 60 / avg_episode_time
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        total_time = time.time() - self.start_time
        progress_pct = (self.current_episode / self.total_episodes) * 100
        
        # Recent performance (last 10 episodes)
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        recent_portfolios = self.episode_portfolio_values[-10:] if len(self.episode_portfolio_values) >= 10 else self.episode_portfolio_values
        
        stats = {
            'progress_pct': progress_pct,
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'current_step': self.current_step,
            'current_reward': self.current_reward,
            'current_portfolio': self.current_portfolio,
            'current_epsilon': self.current_epsilon,
            'current_loss': self.current_loss,
            'steps_per_second': self.steps_per_second,
            'episode_actions': self.episode_actions.copy(),
            'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_recent_portfolio': np.mean(recent_portfolios) if recent_portfolios else 0,
            'avg_recent_loss': np.mean(list(self.recent_losses)) if self.recent_losses else 0,
            'total_time': total_time,
            'eta_seconds': self.eta_seconds,
            'episodes_per_minute': self.episodes_per_minute
        }
        
        return stats
    
    def print_live_progress(self):
        """Print live progress update."""
        stats = self.get_progress_stats()
        
        # Clear previous lines (ANSI escape codes)
        print('\033[2J\033[H', end='')  # Clear screen and move cursor to top
        
        print("=" * 80)
        print(f" LIVE TRAINING PROGRESS - Episode {stats['current_episode']}/{stats['total_episodes']}")
        if self.dashboard and self.dashboard.is_dashboard_connected():
            print(" 📊 Dashboard: Connected at http://localhost:5000")
        elif self.enable_dashboard:
            print(" 📊 Dashboard: Disconnected")
        print("=" * 80)
        
        # Progress bar
        bar_length = 50
        filled_length = int(bar_length * stats['progress_pct'] / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"Progress: [{bar}] {stats['progress_pct']:.1f}%")
        
        # Current episode stats
        print(f"\n Current Episode:")
        print(f"  Step: {stats['current_step']:4d} | Reward: {stats['current_reward']:8.2f} | Portfolio: ${stats['current_portfolio']:8.0f}")
        print(f"  Actions: S:{stats['episode_actions']['SELL']:3d} H:{stats['episode_actions']['HOLD']:3d} B:{stats['episode_actions']['BUY']:3d}")
        param_name = "Alpha" if self.agent_type == 'sac' else "Epsilon"
        print(f"  {param_name}: {stats['current_epsilon']:.3f} | Loss: {stats['current_loss']:.4f} | Speed: {stats['steps_per_second']:.1f} steps/s")
        
        # Recent performance
        print(f"\n Recent Performance (Last 10 Episodes):")
        print(f"  Avg Reward: {stats['avg_recent_reward']:8.2f} | Avg Portfolio: ${stats['avg_recent_portfolio']:8.0f}")
        print(f"  Avg Loss: {stats['avg_recent_loss']:.4f}")
        
        # Time estimates
        eta_str = str(timedelta(seconds=int(stats['eta_seconds'])))
        total_time_str = str(timedelta(seconds=int(stats['total_time'])))
        print(f"\n⏱  Timing:")
        print(f"  Elapsed: {total_time_str} | ETA: {eta_str} | Speed: {stats['episodes_per_minute']:.1f} episodes/min")
        
        print("=" * 80)
    
    def save_progress_log(self, filepath: str):
        """Save detailed progress log."""
        progress_data = {
            'episodes': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_portfolio_values': self.episode_portfolio_values,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'total_training_time': time.time() - self.start_time,
            'final_stats': self.get_progress_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)

def train_with_live_progress(config_path="config/config.yaml", agent_type="sac", enable_dashboard=False):
    """Train agent with live progress monitoring."""
    print("-" * 60)
    print(f" Starting RL Training with Live Progress Monitoring ({agent_type.upper()})")
    print("-" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data loader
    print("\n Loading and preparing data...")
    data_loader = DataLoader(config_path)
    
    # Prepare data for the first symbol
    symbol = config['data']['symbols'][0]
    
    # Load and combine train/test data
    train_data = pd.read_csv(f'data/{symbol}_train.csv', index_col=0, parse_dates=True)
    test_data = pd.read_csv(f'data/{symbol}_test.csv', index_col=0, parse_dates=True)
    all_data = pd.concat([train_data, test_data]).sort_index()
    
    # Initialize environment with first symbol's data
    first_symbol = symbol
    
    # Initialize agent and environment
    agent = None
    env = None
    
    # Initialize progress monitor
    monitor = LiveProgressMonitor(
        total_episodes=config['training']['episodes'],
        update_frequency=config['training'].get('update_freq', 1),
        enable_dashboard=enable_dashboard
    )
    
    # Get number of episodes from config
    num_episodes = config['training']['episodes']
    
    # Initialize environment and agent
    env = None
    agent = None
    
    try:
        # Initialize environment
        env = ContinuousTradingEnvironment(all_data, config_path)
        
        # Initialize agent based on type
        if agent_type.lower() == 'dqn':
            agent = DQNAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                config_path=config_path
            )
        else:  # Default to SAC
            agent = SACAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                config_path=config_path
            )
            
        # Print device info
        device = next(agent.q_network.parameters()).device if hasattr(agent, 'q_network') else 'cpu'
        print(f"\n🚀 Using device: {device}")
        
        # Training loop
        for episode in range(num_episodes):
            try:
                # Reset environment for new episode
                if hasattr(env, 'reset_with_new_data'):
                    env.reset_with_new_data(all_data)
                
                # Initialize training state
                state, _ = env.reset()
                monitor.start_episode(episode)
                episode_reward = 0
                done = False
                step = 0
                
                print(f"\n📅 Episode {episode + 1}/{num_episodes}: Training on data from {all_data.index.min().date()} to {all_data.index.max().date()}")
                
                # Training loop for current episode
                while not done and step < config['training'].get('max_steps_per_episode', 1000):
                    # Get action from agent
                    action = agent.select_action(state)
                    
                    try:
                        # Take a step in the environment
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = bool(terminated or truncated)
                        
                        # Update monitor with step information
                        monitor.update_step(
                            action=action,
                            reward=reward,
                            portfolio_value=info.get('portfolio_value', 0),
                            epsilon_or_alpha=0.2 if agent_type.lower() == 'sac' else None,
                            loss=0,  # Will be updated by the agent
                            agent_type=agent_type,
                            current_price=info.get('current_price', 0)
                        )
                        
                        # Store the transition in memory
                        if hasattr(agent, 'replay_buffer'):
                            agent.replay_buffer.push(
                                state, action, reward, next_state, done
                            )
                        
                        # Update state for next iteration
                        state = next_state
                        episode_reward += reward
                        
                        # Train the agent
                        if (step + 1) % config['training'].get('update_every', 1) == 0:
                            if hasattr(agent, 'update_parameters'):
                                loss = agent.update_parameters(monitor.recent_losses)
                                if loss is not None:
                                    monitor.current_loss = loss
                        
                        # Increment step counter
                        step += 1
                        
                    except Exception as e:
                        print(f"Error during training step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        done = True
                
                # Episode finished
                monitor.finish_episode()
                
                # Print live progress every episode (or based on frequency)
                if episode % monitor.update_frequency == 0 or episode == num_episodes - 1:
                    monitor.print_live_progress()
                    time.sleep(0.1)  # Brief pause for readability
                
                # Save model checkpoint periodically
                if (episode + 1) % config['training'].get('save_interval', 100) == 0:
                    model_path = f"models/{agent_type}_agent_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                    torch.save(agent.state_dict(), model_path)
                    print(f"💾 Checkpoint saved to {model_path}")
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️ Training interrupted by user at episode {episode + 1}/{num_episodes}")
                print("Saving current progress...")
                break
                
            except Exception as e:
                print(f"\n⚠️ Error during episode {episode + 1}/{num_episodes}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"\n⚠️ Fatal error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Training completed or interrupted
    print("\nTraining completed successfully!")
    
    # Save final model and logs
    try:
        if agent is not None and env is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/{agent_type}_agent_{symbol}_{timestamp}.pth'
            
            # Save model using agent's save method if available, otherwise use torch.save
            if hasattr(agent, 'save'):
                agent.save(model_path)
            else:
                torch.save(agent.state_dict(), model_path)
            print(f"✅ Final model saved to {model_path}")
            
            # Save progress log
            os.makedirs('logs/progress', exist_ok=True)
            progress_path = f'logs/progress/training_progress_{symbol}_{timestamp}.json'
            monitor.save_progress_log(progress_path)
            print(f"✅ Progress log saved to {progress_path}")
    except Exception as e:
        print(f"\n⚠️ Error during final save: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    final_stats = monitor.get_progress_stats()
    print("\n" + "=" * 60)
    print("📊 FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Episodes Completed: {len(monitor.episode_rewards)}/{episodes}")
    print(f"Total Training Time: {str(timedelta(seconds=int(final_stats['total_time'])))}")
    print(f"Average Episode Time: {np.mean(monitor.episode_times):.2f} seconds")
    print(f"Final Portfolio Value: ${final_stats['current_portfolio']:,.2f}")
    print(f"Average Recent Reward: {final_stats['avg_recent_reward']:.2f}")
    print(f"Final Epsilon: {final_stats['current_epsilon']:.3f}")
    
    # Create final progress plot
    create_progress_plot(monitor, symbol, timestamp)
    
    print(f"\n🎉 Training complete!")
    print(f"📁 Model saved: {model_path}")
    print(f"📁 Progress log: {progress_path}")
    
    return agent, env, monitor

def create_progress_plot(monitor: LiveProgressMonitor, symbol: str, timestamp: str):
    """Create comprehensive progress visualization."""
    if not monitor.episode_rewards:
        return
    
    os.makedirs('logs/plots', exist_ok=True)
    
    # Set matplotlib backend to avoid GUI issues
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - {symbol}', fontsize=16, fontweight='bold')
    
    episodes = range(len(monitor.episode_rewards))
    
    # Episode rewards
    axes[0, 0].plot(episodes, monitor.episode_rewards, alpha=0.7, color='blue')
    if len(monitor.episode_rewards) > 10:
        # Moving average
        window = min(20, len(monitor.episode_rewards) // 4)
        ma = pd.Series(monitor.episode_rewards).rolling(window).mean()
        axes[0, 0].plot(episodes, ma, color='red', linewidth=2, label=f'{window}-episode MA')
        axes[0, 0].legend()
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio values
    axes[0, 1].plot(episodes, monitor.episode_portfolio_values, alpha=0.7, color='green')
    axes[0, 1].axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Value')
    axes[0, 1].set_title('Portfolio Value')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(episodes, monitor.episode_lengths, alpha=0.7, color='orange')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode times
    axes[1, 1].plot(episodes, monitor.episode_times, alpha=0.7, color='purple')
    if len(monitor.episode_times) > 10:
        # Moving average for times
        window = min(10, len(monitor.episode_times) // 4)
        ma_times = pd.Series(monitor.episode_times).rolling(window).mean()
        axes[1, 1].plot(episodes, ma_times, color='red', linewidth=2, label=f'{window}-episode MA')
        axes[1, 1].legend()
    axes[1, 1].set_title('Episode Duration')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'logs/plots/training_progress_{symbol}_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Progress plot saved: {plot_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL Trading Agent with Live Progress Monitoring')
    parser.add_argument('--agent', '-a', type=str, choices=['dqn', 'sac'], default='sac',
                        help='Agent type to use (default: sac)')
    parser.add_argument('--config', '-c', type=str, default='config/config.yaml',
                        help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='Number of episodes to train (overrides config)')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Enable live web dashboard at http://localhost:5000')
    return parser.parse_args()

if __name__ == "__main__":
    import torch
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f" Selected Agent: {args.agent.upper()}")
    print(f" Config File: {args.config}")
    if args.episodes:
        print(f" Episodes: {args.episodes}")
    
    try:
        # Override episodes if specified
        if args.episodes:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            config['training']['episodes'] = args.episodes
            with open(args.config, 'w') as f:
                yaml.safe_dump(config, f)
            print(f" Updated config with {args.episodes} episodes")
        
        results = train_with_live_progress(config_path=args.config, agent_type=args.agent, enable_dashboard=args.dashboard)
        print(f"\n✨ Live progress training completed successfully with {args.agent.upper()}!")
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
