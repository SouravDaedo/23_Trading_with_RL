#!/usr/bin/env python3
"""
Unified Training Script for RL Trading Agents
Supports both DQN (discrete actions) and SAC (continuous actions) agents.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add src to path
sys.path.append('src')

from data.data_loader import DataLoader
from environment.trading_env import TradingEnvironment
from environment.continuous_trading_env import ContinuousTradingEnvironment, MultiActionContinuousEnvironment
from agents.dqn_agent import DQNAgent
from agents.sac_agent import SACAgent
from utils.metrics import TradingMetrics
from visualization.plotting import TradingVisualizer

def train_dqn_agent(config_path="config/config.yaml", symbol="AAPL"):
    """Train DQN agent with discrete actions."""
    print("=" * 60)
    print(" Training DQN Agent (Discrete Actions)")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load and prepare data
    data_loader = DataLoader(config_path)
    train_data, test_data = data_loader.prepare_data(symbol)
    
    # Create environment
    env = TradingEnvironment(train_data, config_path)
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config_path=config_path
    )
    
    # Training parameters
    episodes = config['training']['episodes']
    
    # Training loop
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Record episode results
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(env.portfolio_value)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            stats = agent.get_training_stats()
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Portfolio: ${avg_portfolio:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Memory Size: {stats['memory_size']}")
            print(f"  Avg Loss: {stats['avg_loss']:.6f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/dqn_{symbol}_{timestamp}.pth"
    agent.save(model_path)
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_portfolio_values': episode_portfolio_values,
        'model_path': model_path,
        'final_portfolio_value': env.portfolio_value
    }

def train_sac_agent(config_path="config/config.yaml", symbol="AAPL", multi_action=False):
    """Train SAC agent with continuous actions."""
    print("=" * 60)
    print(f"🚀 Training SAC Agent ({'Multi-Action' if multi_action else 'Single-Action'} Continuous)")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load and prepare data
    data_loader = DataLoader(config_path)
    train_data, test_data = data_loader.prepare_data(symbol)
    
    # Create continuous environment
    if multi_action:
        env = MultiActionContinuousEnvironment(train_data, config_path)
    else:
        env = ContinuousTradingEnvironment(train_data, config_path)
    
    # Create SAC agent
    agent = SACAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        config_path=config_path
    )
    
    # Training parameters
    episodes = config['training']['episodes']
    
    # Training loop
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.memory) > agent.batch_size:
                update_info = agent.update_parameters()
            
            # Update state
            state = next_state
            total_reward += reward
            step_count += 1
        
        # Record episode results
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(env.portfolio_value)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            stats = agent.get_training_stats()
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Portfolio: ${avg_portfolio:.2f}")
            print(f"  Alpha: {stats['alpha']:.4f}")
            print(f"  Memory Size: {stats['memory_size']}")
            print(f"  Actor Loss: {stats['avg_actor_loss']:.6f}")
            print(f"  Critic Loss: {stats['avg_critic1_loss']:.6f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/sac_{symbol}_{timestamp}.pth"
    agent.save(model_path)
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_portfolio_values': episode_portfolio_values,
        'model_path': model_path,
        'final_portfolio_value': env.portfolio_value
    }

def compare_agents(dqn_results, sac_results, symbol):
    """Compare DQN and SAC agent performance."""
    print("\n" + "=" * 60)
    print("📊 AGENT COMPARISON")
    print("=" * 60)
    
    # Performance comparison
    dqn_final = dqn_results['final_portfolio_value']
    sac_final = sac_results['final_portfolio_value']
    initial_balance = 10000  # From config
    
    dqn_return = (dqn_final - initial_balance) / initial_balance * 100
    sac_return = (sac_final - initial_balance) / initial_balance * 100
    
    print(f"Symbol: {symbol}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print()
    print(f"DQN Agent:")
    print(f"  Final Portfolio: ${dqn_final:,.2f}")
    print(f"  Total Return: {dqn_return:.2f}%")
    print(f"  Avg Episode Reward: {np.mean(dqn_results['episode_rewards']):.4f}")
    print()
    print(f"SAC Agent:")
    print(f"  Final Portfolio: ${sac_final:,.2f}")
    print(f"  Total Return: {sac_return:.2f}%")
    print(f"  Avg Episode Reward: {np.mean(sac_results['episode_rewards']):.4f}")
    print()
    
    if sac_return > dqn_return:
        print(f"🏆 SAC Agent outperformed DQN by {sac_return - dqn_return:.2f}%")
    else:
        print(f"🏆 DQN Agent outperformed SAC by {dqn_return - sac_return:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Rewards comparison
    plt.subplot(2, 2, 1)
    plt.plot(dqn_results['episode_rewards'], label='DQN', alpha=0.7)
    plt.plot(sac_results['episode_rewards'], label='SAC', alpha=0.7)
    plt.title('Episode Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Portfolio values comparison
    plt.subplot(2, 2, 2)
    plt.plot(dqn_results['episode_portfolio_values'], label='DQN', alpha=0.7)
    plt.plot(sac_results['episode_portfolio_values'], label='SAC', alpha=0.7)
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Smoothed rewards
    plt.subplot(2, 2, 3)
    window = 50
    dqn_smooth = np.convolve(dqn_results['episode_rewards'], np.ones(window)/window, mode='valid')
    sac_smooth = np.convolve(sac_results['episode_rewards'], np.ones(window)/window, mode='valid')
    plt.plot(dqn_smooth, label='DQN (smoothed)', linewidth=2)
    plt.plot(sac_smooth, label='SAC (smoothed)', linewidth=2)
    plt.title(f'Smoothed Rewards (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Final comparison bar chart
    plt.subplot(2, 2, 4)
    agents = ['DQN', 'SAC']
    returns = [dqn_return, sac_return]
    colors = ['blue', 'orange']
    bars = plt.bar(agents, returns, color=colors, alpha=0.7)
    plt.title('Final Returns Comparison')
    plt.ylabel('Return (%)')
    plt.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{return_val:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/agent_comparison_{symbol}_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to: {plot_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument("--agent", choices=["dqn", "sac", "both"], default="both",
                       help="Which agent to train")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to trade")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--multi-action", action="store_true", 
                       help="Use multi-action continuous environment for SAC")
    
    args = parser.parse_args()
    
    print(f"🎯 Training Configuration:")
    print(f"  Agent: {args.agent}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Config: {args.config}")
    print(f"  Multi-action: {args.multi_action}")
    print()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    results = {}
    
    try:
        if args.agent in ["dqn", "both"]:
            print("Starting DQN training...")
            results['dqn'] = train_dqn_agent(args.config, args.symbol)
            print("✅ DQN training completed!")
        
        if args.agent in ["sac", "both"]:
            print("Starting SAC training...")
            results['sac'] = train_sac_agent(args.config, args.symbol, args.multi_action)
            print("✅ SAC training completed!")
        
        # Compare agents if both were trained
        if args.agent == "both":
            compare_agents(results['dqn'], results['sac'], args.symbol)
        
        print("\n✨ Training pipeline completed successfully!")
        
        # Print model paths
        print("\n📁 Saved Models:")
        for agent_type, result in results.items():
            print(f"  {agent_type.upper()}: {result['model_path']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
