#!/usr/bin/env python3
"""
Enhanced training script with detailed action tracking and progress monitoring.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import defaultdict, deque

# Add src to path
sys.path.append('src')

from data.data_loader import DataLoader
from environment.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from utils.metrics import TradingMetrics
from visualization.plotting import TradingVisualizer

class EpisodeTracker:
    """Track detailed episode metrics and agent actions."""
    
    def __init__(self):
        self.reset_episode()
        self.episode_history = []
        self.action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
    def reset_episode(self):
        """Reset tracking for new episode."""
        self.episode_data = {
            'episode': 0,
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'positions': [],
            'prices': [],
            'q_values': [],
            'epsilon': 0,
            'trades': [],
            'step_details': []
        }
        
    def log_step(self, step, state, action, reward, next_state, info, q_values=None, epsilon=None):
        """Log detailed step information."""
        step_detail = {
            'step': step,
            'action': action,
            'action_name': self.action_names[action],
            'reward': reward,
            'portfolio_value': info.get('portfolio_value', 0),
            'position': info.get('position', 0),
            'cash': info.get('cash', 0),
            'stock_value': info.get('stock_value', 0),
            'price': info.get('current_price', 0),
            'total_trades': info.get('total_trades', 0)
        }
        
        if q_values is not None:
            step_detail['q_values'] = q_values.tolist() if hasattr(q_values, 'tolist') else q_values
            step_detail['max_q_value'] = float(np.max(q_values))
            step_detail['action_confidence'] = float(np.max(q_values) - np.mean(q_values))
            
        if epsilon is not None:
            step_detail['epsilon'] = epsilon
            
        # Track trades
        if len(self.episode_data['step_details']) > 0:
            prev_trades = self.episode_data['step_details'][-1]['total_trades']
            if step_detail['total_trades'] > prev_trades:
                trade_info = {
                    'step': step,
                    'action': self.action_names[action],
                    'price': step_detail['price'],
                    'portfolio_value': step_detail['portfolio_value']
                }
                self.episode_data['trades'].append(trade_info)
        
        self.episode_data['step_details'].append(step_detail)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        self.episode_data['portfolio_values'].append(step_detail['portfolio_value'])
        self.episode_data['positions'].append(step_detail['position'])
        self.episode_data['prices'].append(step_detail['price'])
        
        if q_values is not None:
            self.episode_data['q_values'].append(step_detail.get('max_q_value', 0))
        if epsilon is not None:
            self.episode_data['epsilon'] = epsilon
    
    def finalize_episode(self, episode_num, total_reward, final_portfolio_value):
        """Finalize episode data and calculate metrics."""
        self.episode_data['episode'] = episode_num
        self.episode_data['total_reward'] = total_reward
        self.episode_data['final_portfolio_value'] = final_portfolio_value
        self.episode_data['total_steps'] = len(self.episode_data['actions'])
        
        # Calculate episode metrics
        actions_array = np.array(self.episode_data['actions'])
        self.episode_data['action_counts'] = {
            'HOLD': int(np.sum(actions_array == 0)),
            'BUY': int(np.sum(actions_array == 1)),
            'SELL': int(np.sum(actions_array == 2))
        }
        
        self.episode_data['action_percentages'] = {
            action: count / len(actions_array) * 100 
            for action, count in self.episode_data['action_counts'].items()
        }
        
        # Portfolio performance
        portfolio_values = self.episode_data['portfolio_values']
        if len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            self.episode_data['return_pct'] = (final_portfolio_value - initial_value) / initial_value * 100
            self.episode_data['max_portfolio'] = max(portfolio_values)
            self.episode_data['min_portfolio'] = min(portfolio_values)
            self.episode_data['volatility'] = np.std(portfolio_values)
        
        # Trading metrics
        self.episode_data['total_trades'] = len(self.episode_data['trades'])
        self.episode_data['avg_reward_per_step'] = total_reward / len(self.episode_data['actions']) if self.episode_data['actions'] else 0
        
        # Store episode
        self.episode_history.append(self.episode_data.copy())
        
    def get_episode_summary(self, episode_num):
        """Get summary of specific episode."""
        if episode_num < len(self.episode_history):
            return self.episode_history[episode_num]
        return None
    
    def get_recent_episodes_summary(self, n=10):
        """Get summary of last n episodes."""
        recent = self.episode_history[-n:] if len(self.episode_history) >= n else self.episode_history
        
        if not recent:
            return {}
            
        summary = {
            'episodes_count': len(recent),
            'avg_total_reward': np.mean([ep['total_reward'] for ep in recent]),
            'avg_return_pct': np.mean([ep.get('return_pct', 0) for ep in recent]),
            'avg_trades_per_episode': np.mean([ep['total_trades'] for ep in recent]),
            'avg_steps_per_episode': np.mean([ep['total_steps'] for ep in recent]),
            'action_distribution': {}
        }
        
        # Aggregate action counts
        total_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for episode in recent:
            for action, count in episode['action_counts'].items():
                total_actions[action] += count
        
        total_count = sum(total_actions.values())
        summary['action_distribution'] = {
            action: count / total_count * 100 for action, count in total_actions.items()
        }
        
        return summary
    
    def save_tracking_data(self, filepath):
        """Save all tracking data to file."""
        tracking_data = {
            'episode_history': self.episode_history,
            'summary': self.get_recent_episodes_summary(len(self.episode_history))
        }
        
        with open(filepath, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f" Tracking data saved to: {filepath}")

def train_agent_with_tracking(config_path="config/config.yaml"):
    """Train the RL trading agent with detailed tracking."""
    print("-" * 60)
    print(" Starting RL Trading Agent Training with Tracking")
    print("-" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize tracker
    tracker = EpisodeTracker()
    
    # Initialize data loader
    print("\n Loading and preparing data...")
    data_loader = DataLoader(config_path)
    
    # Prepare data for the first symbol
    symbol = config['data']['symbols'][0]
    train_data, test_data = data_loader.prepare_data(symbol)
    
    # Save processed data
    data_loader.save_data(train_data, f'{symbol}_train.csv')
    data_loader.save_data(test_data, f'{symbol}_test.csv')
    
    # Initialize environment
    print(f"\n🏗️ Setting up trading environment for {symbol}...")
    env = TradingEnvironment(train_data, config_path)
    
    # Initialize agent
    print("\n🤖 Initializing DQN agent...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config_path)
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {agent.device}")
    
    # Training parameters
    episodes = config['training']['episodes']
    
    # Training metrics
    episode_rewards = []
    episode_portfolio_values = []
    episode_lengths = []
    
    print(f"\n🎯 Starting training for {episodes} episodes...")
    print("-" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        tracker.reset_episode()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action with Q-values
            action = agent.act(state, training=True)
            
            # Get Q-values for tracking
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).cpu().numpy()[0]
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Log detailed step information
            tracker.log_step(
                step=steps,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                info=info,
                q_values=q_values,
                epsilon=agent.epsilon
            )
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Finalize episode tracking
        tracker.finalize_episode(episode, total_reward, info['portfolio_value'])
        
        # Store episode metrics
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        episode_lengths.append(steps)
        
        # Print detailed progress
        if episode % 10 == 0:
            recent_summary = tracker.get_recent_episodes_summary(10)
            
            print(f"\n📈 Episode {episode:4d} Summary:")
            print(f"  Reward: {total_reward:8.2f} | Portfolio: ${info['portfolio_value']:8.0f}")
            print(f"  Actions: H:{tracker.episode_data['action_counts']['HOLD']:2d} "
                  f"B:{tracker.episode_data['action_counts']['BUY']:2d} "
                  f"S:{tracker.episode_data['action_counts']['SELL']:2d} | "
                  f"Trades: {tracker.episode_data['total_trades']:2d}")
            print(f"  Return: {tracker.episode_data.get('return_pct', 0):6.2f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            if recent_summary:
                print(f"  📊 Last 10 Episodes Avg:")
                print(f"    Reward: {recent_summary['avg_total_reward']:8.2f} | "
                      f"Return: {recent_summary['avg_return_pct']:6.2f}%")
                print(f"    Actions: H:{recent_summary['action_distribution']['HOLD']:4.1f}% "
                      f"B:{recent_summary['action_distribution']['BUY']:4.1f}% "
                      f"S:{recent_summary['action_distribution']['SELL']:4.1f}%")
    
    print("\n✅ Training completed!")
    
    # Save trained model
    print("\n💾 Saving trained model...")
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/dqn_agent_{symbol}_{timestamp}.pth'
    agent.save(model_path)
    
    # Save tracking data
    os.makedirs('logs/tracking', exist_ok=True)
    tracking_path = f'logs/tracking/episode_tracking_{symbol}_{timestamp}.json'
    tracker.save_tracking_data(tracking_path)
    
    # Generate enhanced visualizations
    create_tracking_visualizations(tracker, symbol, timestamp)
    
    # Final summary
    final_summary = tracker.get_recent_episodes_summary(episodes)
    print("\n" + "=" * 60)
    print("📊 FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {episodes}")
    print(f"Average Reward: {final_summary['avg_total_reward']:.2f}")
    print(f"Average Return: {final_summary['avg_return_pct']:.2f}%")
    print(f"Average Trades per Episode: {final_summary['avg_trades_per_episode']:.1f}")
    print(f"Action Distribution:")
    for action, pct in final_summary['action_distribution'].items():
        print(f"  {action}: {pct:.1f}%")
    
    return agent, env, tracker

def create_tracking_visualizations(tracker, symbol, timestamp):
    """Create comprehensive visualizations from tracking data."""
    print("\n📊 Creating tracking visualizations...")
    
    os.makedirs('logs/plots', exist_ok=True)
    
    # Prepare data
    episodes_data = tracker.episode_history
    if not episodes_data:
        print("No episode data to visualize")
        return
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Agent Training Analysis - {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Portfolio value over episodes
    portfolio_values = [ep['final_portfolio_value'] for ep in episodes_data]
    episodes_nums = [ep['episode'] for ep in episodes_data]
    
    axes[0, 0].plot(episodes_nums, portfolio_values, alpha=0.7, color='green')
    axes[0, 0].axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Value')
    axes[0, 0].set_title('Portfolio Value per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Action distribution over time
    window_size = max(10, len(episodes_data) // 20)
    hold_pcts = []
    buy_pcts = []
    sell_pcts = []
    
    for i in range(len(episodes_data)):
        start_idx = max(0, i - window_size + 1)
        window_episodes = episodes_data[start_idx:i+1]
        
        total_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for ep in window_episodes:
            for action, count in ep['action_counts'].items():
                total_actions[action] += count
        
        total_count = sum(total_actions.values())
        if total_count > 0:
            hold_pcts.append(total_actions['HOLD'] / total_count * 100)
            buy_pcts.append(total_actions['BUY'] / total_count * 100)
            sell_pcts.append(total_actions['SELL'] / total_count * 100)
        else:
            hold_pcts.append(0)
            buy_pcts.append(0)
            sell_pcts.append(0)
    
    axes[0, 1].plot(episodes_nums, hold_pcts, label='HOLD', alpha=0.8)
    axes[0, 1].plot(episodes_nums, buy_pcts, label='BUY', alpha=0.8)
    axes[0, 1].plot(episodes_nums, sell_pcts, label='SELL', alpha=0.8)
    axes[0, 1].set_title(f'Action Distribution (Rolling {window_size}-episode window)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Action Percentage (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Trades per episode
    trades_per_episode = [ep['total_trades'] for ep in episodes_data]
    axes[1, 0].bar(episodes_nums, trades_per_episode, alpha=0.7, color='orange')
    axes[1, 0].set_title('Trades per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Return percentage distribution
    returns = [ep.get('return_pct', 0) for ep in episodes_data]
    axes[1, 1].hist(returns, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    axes[1, 1].set_title('Episode Return Distribution')
    axes[1, 1].set_xlabel('Return (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Cumulative rewards
    rewards = [ep['total_reward'] for ep in episodes_data]
    cumulative_rewards = np.cumsum(rewards)
    axes[2, 0].plot(episodes_nums, cumulative_rewards, alpha=0.8, color='blue')
    axes[2, 0].set_title('Cumulative Rewards')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Cumulative Reward')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Action confidence (Q-value spread) over time
    if any('q_values' in ep and ep['q_values'] for ep in episodes_data):
        avg_q_values = []
        for ep in episodes_data:
            if ep.get('q_values'):
                avg_q_values.append(np.mean(ep['q_values']))
            else:
                avg_q_values.append(0)
        
        axes[2, 1].plot(episodes_nums, avg_q_values, alpha=0.8, color='red')
        axes[2, 1].set_title('Average Q-Values per Episode')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Average Q-Value')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'Q-Values not available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Q-Values Analysis')
    
    plt.tight_layout()
    plot_path = f'logs/plots/training_tracking_{symbol}_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Tracking plots saved to: {plot_path}")

if __name__ == "__main__":
    import torch
    
    try:
        results = train_agent_with_tracking()
        print("\n✨ Enhanced training pipeline executed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
