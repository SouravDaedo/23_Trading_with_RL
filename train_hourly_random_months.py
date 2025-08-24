#!/usr/bin/env python3
"""
Train agent on hourly data with random monthly periods.
Each episode uses data from a randomly selected month.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import json
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append('src')

from data.data_loader import DataLoader
from environment.continuous_trading_env import ContinuousTradingEnvironment
from agents.sac_agent import SACAgent
from agents.dqn_agent import DQNAgent

# Dashboard integration
try:
    from dashboard.dashboard_integration import DashboardConnector
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️  Dashboard integration not available. Install flask and flask-socketio to enable.")

class MonthlyEpisodeTrainer:
    """Trainer for monthly episodic training on hourly data."""
    
    def __init__(self, config_path: str = "config/config.yaml", agent_type: str = "sac", enable_dashboard: bool = False):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        self.agent_type = agent_type.lower()
        self.enable_dashboard = enable_dashboard and DASHBOARD_AVAILABLE
        
        # Initialize dashboard if enabled
        self.dashboard = None
        if self.enable_dashboard:
            self.dashboard = DashboardConnector()
            print("📊 Dashboard connector initialized")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize data loader
        self.data_loader = DataLoader(config_path)
        self.symbol = self.data_loader.symbols[0]  # Use first symbol
        
        # Load all historical data
        print("Loading historical data...")
        self.full_data = self.data_loader.load_data(self.symbol)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_portfolio_values = []
        self.episode_lengths = []
        self.episode_times = []
        self.start_time = time.time()
        self.action_counts = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
        
        # Initialize environment and agent
        self.env = None
        self.agent = None
        
    def get_random_monthly_period(self, min_days: int = 14, max_days: int = 30) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Select a random monthly period from the available data."""
        if len(self.full_data) < min_days * 24:  # Assuming hourly data
            return self.full_data.index[0], self.full_data.index[-1]
        
        # Get all unique months in the data
        months = sorted(set((d.year, d.month) for d in self.full_data.index))
        if not months:
            return self.full_data.index[0], self.full_data.index[-1]
        
        # Select a random month
        year, month = random.choice(months)
        
        # Get all data for the selected month
        month_mask = (self.full_data.index.year == year) & (self.full_data.index.month == month)
        month_data = self.full_data[month_mask]
        
        if len(month_data) < min_days * 24:
            return month_data.index[0], month_data.index[-1]
        
        # Select a random start date within the month
        max_start = len(month_data) - min_days * 24
        if max_start <= 0:
            return month_data.index[0], month_data.index[-1]
        
        start_idx = random.randint(0, max_start)
        start_date = month_data.index[start_idx]
        
        # Select end date (between min_days and max_days after start)
        max_end_idx = min(start_idx + max_days * 24, len(month_data) - 1)
        min_end_idx = min(start_idx + min_days * 24, max_end_idx)
        
        if min_end_idx >= max_end_idx:
            end_date = month_data.index[-1]
        else:
            end_idx = random.randint(min_end_idx, max_end_idx)
            end_date = month_data.index[end_idx]
        
        return start_date, end_date
    
    def initialize_agent(self, state_dim: int, action_dim: int):
        """Initialize the RL agent."""
        if self.agent_type == "dqn":
            return DQNAgent(
                state_size=state_dim,
                action_size=action_dim,
                config_path=self.config_path
            )
        else:  # Default to SAC
            return SACAgent(
                state_size=state_dim,
                action_size=action_dim,
                config_path=self.config_path
            )
    
    def _update_dashboard(self, episode: int, total_episodes: int, current_step: int, 
                        total_steps: int, reward: float, portfolio_value: float, 
                        loss: Optional[float] = None, alpha: float = 0.2, 
                        epsilon: float = 0.0):
        """Update the dashboard with current training metrics."""
        if not self.enable_dashboard or self.dashboard is None:
            return
            
        # Calculate metrics
        elapsed = time.time() - self.start_time
        steps_per_sec = current_step / (elapsed + 1e-5)
        
        # Prepare update data
        update_data = {
            'episode': episode,
            'total_episodes': total_episodes,
            'step': current_step,
            'total_steps': total_steps,
            'reward': float(reward),
            'portfolio_value': float(portfolio_value),
            'loss': float(loss) if loss is not None else 0.0,
            'alpha': float(alpha),
            'epsilon': float(epsilon),
            'elapsed_time': elapsed,
            'steps_per_sec': steps_per_sec,
            'action_counts': self.action_counts.copy(),
            'episode_rewards': [float(r) for r in self.episode_rewards[-10:]],
            'episode_portfolio_values': [float(v) for v in self.episode_portfolio_values[-10:]],
            'episode_losses': [float(l) for l in self.episode_losses[-10:]] if self.episode_losses else []
        }
        
        # Send update to dashboard
        try:
            self.dashboard.update(update_data)
        except Exception as e:
            print(f"⚠️  Dashboard update failed: {str(e)}")
    
    def _update_action_counts(self, action: np.ndarray):
        """Update action counts for dashboard display."""
        if self.agent_type == 'dqn':
            # Discrete action space
            if action == 0:
                self.action_counts['SELL'] += 1
            elif action == 1:
                self.action_counts['HOLD'] += 1
            elif action == 2:
                self.action_counts['BUY'] += 1
        else:
            # Continuous action space (SAC)
            if action < -0.5:
                self.action_counts['SELL'] += 1
            elif action > 0.5:
                self.action_counts['BUY'] += 1
            else:
                self.action_counts['HOLD'] += 1
    
    def train(self, episodes: int = 100, save_interval: int = 10):
        """Train the agent on random monthly periods."""
        print(f"Starting training on {self.symbol} with {self.agent_type.upper()} agent")
        if self.enable_dashboard:
            print(f"📊 Dashboard available at: http://localhost:5000")
        
        # Training loop
        for episode in range(1, episodes + 1):
            episode_start_time = time.time()
            episode_losses = []
            # Select a random monthly period
            start_date, end_date = self.get_random_monthly_period()
            episode_data = self.full_data[
                (self.full_data.index >= start_date) & 
                (self.full_data.index <= end_date)
            ].copy()
            
            print(f"\n📅 Episode {episode}/{episodes}: {start_date.date()} to {end_date.date()} "
                  f"({len(episode_data)} hours)")
            
            # Initialize or reset environment with new data
            if self.env is None:
                self.env = ContinuousTradingEnvironment(episode_data, self.config_path)
                self.agent = self.initialize_agent(
                    state_dim=self.env.observation_space.shape[0],
                    action_dim=self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
                )
            else:
                if hasattr(self.env, 'reset_with_new_data'):
                    self.env.reset_with_new_data(episode_data)
                else:
                    self.env = ContinuousTradingEnvironment(episode_data, self.config_path)
            
            # Run episode
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Get action from agent
                action = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.agent.replay_buffer.push(state, action, next_state, reward, done)
                
                # Update agent
                loss = None
                if len(self.agent.replay_buffer) > self.config['training'].get('batch_size', 64):
                    loss = self.agent.update_parameters()
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Update action counts for dashboard
                self._update_action_counts(action)
                
                # Update dashboard
                if step % 10 == 0:  # Update dashboard every 10 steps
                    alpha = getattr(self.agent, 'alpha', 0.2)
                    epsilon = getattr(self.agent, 'epsilon', 0.0)
                    self._update_dashboard(
                        episode=episode,
                        total_episodes=episodes,
                        current_step=step,
                        total_steps=len(episode_data) - 1,
                        reward=episode_reward,
                        portfolio_value=info.get('portfolio_value', 0),
                        loss=np.mean(episode_losses) if episode_losses else None,
                        alpha=alpha,
                        epsilon=epsilon
                    )
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # Calculate episode metrics
            episode_time = time.time() - episode_start_time
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)
            self.episode_portfolio_values.append(info.get('portfolio_value', 0))
            self.episode_lengths.append(self.env.current_step)
            self.episode_times.append(episode_time)
            
            # Print episode summary
            print(f"  Reward: {episode_reward:.2f}, "
                  f"Portfolio: ${info.get('portfolio_value', 0):.2f}, "
                  f"Steps: {self.env.current_step}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Time: {episode_time:.1f}s")
            
            # Final dashboard update for the episode
            if self.enable_dashboard:
                alpha = getattr(self.agent, 'alpha', 0.2)
                epsilon = getattr(self.agent, 'epsilon', 0.0)
                self._update_dashboard(
                    episode=episode,
                    total_episodes=episodes,
                    current_step=step,
                    total_steps=step,
                    reward=episode_reward,
                    portfolio_value=info.get('portfolio_value', 0),
                    loss=avg_loss,
                    alpha=alpha,
                    epsilon=epsilon
                )
            
            # Save model periodically
            if episode % save_interval == 0 or episode == episodes:
                model_path = f"models/{self.agent_type}_agent_{self.symbol}_ep{episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                self.agent.save(model_path)
                print(f"💾 Saved model to {model_path}")
                
                # Save training progress
                progress = {
                    'episode': episode,
                    'rewards': self.episode_rewards,
                    'losses': self.episode_losses,
                    'portfolio_values': self.episode_portfolio_values,
                    'episode_lengths': self.episode_lengths,
                    'episode_times': self.episode_times,
                    'action_counts': self.action_counts,
                    'timestamp': datetime.now().isoformat()
                }
                with open(f'training_progress_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                    json.dump(progress, f)
        
        # Save final model
        final_path = f"models/{self.agent_type}_agent_{self.symbol}_final.pth"
        self.agent.save(final_path)
        print(f"\n✅ Training complete! Final model saved to {final_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL agent on random monthly periods of hourly data')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'dqn'], help='Agent type (sac or dqn)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N episodes')
    parser.add_argument('--dashboard', action='store_true', help='Enable web dashboard')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run trainer
        trainer = MonthlyEpisodeTrainer(
            config_path=args.config, 
            agent_type=args.agent,
            enable_dashboard=args.dashboard
        )
        trainer.train(episodes=args.episodes, save_interval=args.save_interval)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Ensure dashboard is properly closed
        if 'trainer' in locals() and hasattr(trainer, 'dashboard') and trainer.dashboard:
            trainer.dashboard.close()

if __name__ == "__main__":
    main()
