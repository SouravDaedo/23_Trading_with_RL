#!/usr/bin/env python3
"""
Main training script for the RL trading agent.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append('src')

from data.data_loader import DataLoader
from environment.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from utils.metrics import TradingMetrics
from visualization.plotting import TradingVisualizer

def train_agent(config_path="config/config.yaml"):
    """Train the RL trading agent."""
    print("-" * 60)
    print("Starting RL Trading Agent Training")
    print("-" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data loader
    print("\n Loading and preparing data...")
    data_loader = DataLoader(config_path)
    
    # Prepare data for the first symbol (can be extended for multiple symbols)
    symbol = config['data']['symbols'][0]
    train_data, test_data = data_loader.prepare_data(symbol)
    
    # Save processed data
    data_loader.save_data(train_data, f'{symbol}_train.csv')
    data_loader.save_data(test_data, f'{symbol}_test.csv')
    
    # Initialize environment
    print(f"\n🏗️ Setting up trading environment for {symbol}...")
    env = TradingEnvironment(train_data, config_path)
    
    # Initialize agent
    print("\n Initializing DQN agent...")
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
    
    print(f"\n Starting training for {episodes} episodes...")
    print("-" * 60)
    ep_int = 10

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
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
        
        # Store episode metrics
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        episode_lengths.append(steps)
        
        # Print progress
        if episode % ep_int == 0:
            avg_reward = np.mean(episode_rewards[-ep_int:])
            avg_portfolio = np.mean(episode_portfolio_values[-ep_int:])
            stats = agent.get_training_stats()
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Portfolio: ${avg_portfolio:8.0f} | "
                  f"Epsilon: {stats['epsilon']:.3f} | "
                  f"Loss: {stats['avg_loss']:.4f}")
    
    print("\n Training completed!")
    
    # Save trained model
    print("\n Saving trained model...")
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/dqn_agent_{symbol}_{timestamp}.pth'
    agent.save(model_path)
    
    # Calculate final performance metrics
    print("\n📈 Calculating performance metrics...")
    portfolio_stats = env.get_portfolio_stats()
    
    print("\n" + "=" * 60)
    print("📊 TRAINING RESULTS")
    print("=" * 60)
    print(f"Final Portfolio Value: ${portfolio_stats['final_portfolio_value']:,.2f}")
    print(f"Total Return: {portfolio_stats['total_return']:.2f}%")
    print(f"Total Trades: {portfolio_stats['total_trades']}")
    print(f"Win Rate: {portfolio_stats['win_rate']:.2f}%")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Training Steps: {agent.training_step}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = TradingVisualizer()
    
    # Plot training metrics
    os.makedirs('logs/plots', exist_ok=True)
    
    # Training performance
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(episode_portfolio_values)
    plt.title('Portfolio Value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if agent.losses:
        plt.plot(agent.losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if agent.q_values:
        plt.plot(agent.q_values)
        plt.title('Average Q-Values')
        plt.xlabel('Training Step')
        plt.ylabel('Q-Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'logs/plots/training_summary_{symbol}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training complete! Model saved to: {model_path}")
    print("📁 Check the 'logs/plots' directory for visualization outputs.")
    
    return agent, env, {
        'episode_rewards': episode_rewards,
        'episode_portfolio_values': episode_portfolio_values,
        'portfolio_stats': portfolio_stats,
        'model_path': model_path
    }

if __name__ == "__main__":
    try:
        results = train_agent()
        print("\n✨ Training pipeline executed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
