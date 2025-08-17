"""
Multi-Agent Training Pipeline
Train multiple specialized agents for multi-stock trading.
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Dict, List, Any

from src.data.data_loader import DataLoader
from src.agents.multi_agent_system import MultiAgentTradingSystem

def train_multi_agent_system(config_path="config/config.yaml", 
                            symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
                            agent_types=["dqn", "sac", "dqn", "sac"],
                            episodes=1000):
    """
    Train multi-agent trading system.
    
    Args:
        config_path: Path to configuration file
        symbols: List of stock symbols to trade
        agent_types: List of agent types for each stock
        episodes: Number of training episodes
    """
    print("=" * 80)
    print("🚀 Multi-Agent Trading System Training")
    print("=" * 80)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"Training Configuration:")
    print(f"  Symbols: {symbols}")
    print(f"  Agent Types: {agent_types}")
    print(f"  Episodes: {episodes}")
    print(f"  Initial Balance: ${config['environment']['initial_balance']:,.2f}")
    
    # Load data for all symbols
    print("\n Loading market data...")
    data_loader = DataLoader(config_path)
    stock_data = {}
    
    for symbol in symbols:
        print(f"  Loading {symbol}...")
        data = data_loader.load_data(symbol)
        
        if data is None or data.empty:
            print(f"   Failed to load data for {symbol}")
            continue
            
        stock_data[symbol] = data
        print(f"  {symbol}: {len(data)} data points")
    
    if len(stock_data) != len(symbols):
        print("⚠️  Warning: Some symbols failed to load. Continuing with available data.")
        symbols = list(stock_data.keys())
        agent_types = agent_types[:len(symbols)]
    
    # Initialize multi-agent system
    print(f"\n🤖 Initializing Multi-Agent System...")
    mas = MultiAgentTradingSystem(symbols, agent_types, config_path)
    mas.setup_agents(stock_data)
    
    # Training metrics
    episode_rewards = []
    portfolio_values = []
    allocation_entropy = []  # Measure of allocation diversity
    
    # Training loop
    print(f"\n Starting Training...")
    print("-" * 60)
    
    for episode in range(episodes):
        # Train one episode
        episode_results = mas.train_episode()
        
        # Calculate episode metrics
        total_episode_reward = sum(
            sum(rewards) for rewards in episode_results['stock_rewards'].values()
        )
        final_portfolio_value = episode_results['portfolio_values'][-1] if episode_results['portfolio_values'] else 0
        
        # Calculate allocation entropy (diversity measure)
        if episode_results['allocations']:
            avg_allocation = np.mean(episode_results['allocations'], axis=0)
            entropy = -np.sum(avg_allocation * np.log(avg_allocation + 1e-8))
            allocation_entropy.append(entropy)
        else:
            allocation_entropy.append(0)
        
        # Record metrics
        episode_rewards.append(total_episode_reward)
        portfolio_values.append(final_portfolio_value)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(portfolio_values[-100:])
            avg_entropy = np.mean(allocation_entropy[-100:])
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Portfolio: ${avg_portfolio:.2f}")
            print(f"  Allocation Entropy: {avg_entropy:.4f}")
            
            # Individual agent stats
            system_stats = mas.get_system_stats()
            print("  Agent Performance:")
            for symbol, stats in system_stats['stock_agents'].items():
                agent_type = mas.stock_agents[symbol].agent_type.upper()
                print(f"    {symbol} ({agent_type}): Win Rate: {stats['win_rate']:.2%}, "
                      f"Avg Reward: {stats['avg_reward']:.4f}")
            
            # Current allocation
            if system_stats['allocation_history']:
                current_allocation = system_stats['allocation_history'][-1]['allocation']
                print("  Current Allocation:")
                for i, symbol in enumerate(symbols):
                    print(f"    {symbol}: {current_allocation[i]:.2%}")
            print("-" * 60)
    
    # Save trained models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/multi_agent_{timestamp}"
    mas.save_all_agents(model_dir)
    
    # Generate training plots
    print("\n📊 Generating training plots...")
    plot_training_results(episode_rewards, portfolio_values, allocation_entropy, 
                         symbols, model_dir)
    
    # Final evaluation
    print("\n🎯 Final Evaluation...")
    final_stats = mas.get_system_stats()
    
    print("Final Agent Performance:")
    total_return = 0
    for symbol, stats in final_stats['stock_agents'].items():
        agent_type = mas.stock_agents[symbol].agent_type.upper()
        print(f"  {symbol} ({agent_type}):")
        print(f"    Win Rate: {stats['win_rate']:.2%}")
        print(f"    Total Return: {stats['total_return']:.4f}")
        print(f"    Trade Count: {stats['trade_count']}")
        total_return += stats['total_return']
    
    print(f"\nPortfolio Summary:")
    portfolio_metrics = final_stats['portfolio_metrics']
    print(f"  Total Portfolio Value: ${portfolio_metrics['total_value']:.2f}")
    print(f"  Total Return: {portfolio_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
    
    # Save training summary
    training_summary = {
        'symbols': symbols,
        'agent_types': agent_types,
        'episodes': episodes,
        'final_stats': final_stats,
        'training_metrics': {
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values,
            'allocation_entropy': allocation_entropy
        }
    }
    
    summary_path = os.path.join(model_dir, 'training_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(training_summary, f, default_flow_style=False)
    
    print(f"\n Training completed!")
    print(f" Models saved to: {model_dir}")
    print(f" Training summary saved to: {summary_path}")
    
    return mas, training_summary

def plot_training_results(episode_rewards: List[float], 
                         portfolio_values: List[float],
                         allocation_entropy: List[float],
                         symbols: List[str],
                         save_dir: str):
    """Plot training results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Agent Training Results', fontsize=16, fontweight='bold')
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.7, color='blue')
    axes[0, 0].plot(pd.Series(episode_rewards).rolling(100).mean(), 
                    color='red', linewidth=2, label='100-episode MA')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio values
    axes[0, 1].plot(portfolio_values, alpha=0.7, color='green')
    axes[0, 1].plot(pd.Series(portfolio_values).rolling(100).mean(), 
                    color='red', linewidth=2, label='100-episode MA')
    axes[0, 1].set_title('Portfolio Values')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Allocation entropy
    axes[1, 0].plot(allocation_entropy, alpha=0.7, color='purple')
    axes[1, 0].plot(pd.Series(allocation_entropy).rolling(100).mean(), 
                    color='red', linewidth=2, label='100-episode MA')
    axes[1, 0].set_title('Allocation Entropy (Diversity)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Returns comparison
    if len(portfolio_values) > 0:
        initial_value = portfolio_values[0]
        returns = [(pv - initial_value) / initial_value * 100 for pv in portfolio_values]
        axes[1, 1].plot(returns, alpha=0.7, color='orange')
        axes[1, 1].plot(pd.Series(returns).rolling(100).mean(), 
                        color='red', linewidth=2, label='100-episode MA')
        axes[1, 1].set_title('Portfolio Returns (%)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to: {plot_path}")

def evaluate_multi_agent_system(model_dir: str, 
                               symbols: List[str],
                               agent_types: List[str],
                               config_path="config/config.yaml"):
    """
    Evaluate trained multi-agent system.
    
    Args:
        model_dir: Directory containing trained models
        symbols: List of stock symbols
        agent_types: List of agent types
        config_path: Path to configuration file
    """
    print("=" * 60)
    print(" Multi-Agent System Evaluation")
    print("=" * 60)
    
    # Load test data
    data_loader = DataLoader(config_path)
    test_data = {}
    
    for symbol in symbols:
        data = data_loader.load_data(symbol, start_date="2023-01-01")
        if data is not None:
            test_data[symbol] = data
    
    # Initialize and load trained system
    mas = MultiAgentTradingSystem(symbols, agent_types, config_path)
    mas.setup_agents(test_data)
    mas.load_all_agents(model_dir)
    
    # Run evaluation
    results = mas.evaluate_agents(test_data)
    
    print("Evaluation Results:")
    for symbol, result in results.items():
        agent_type = mas.stock_agents[symbol].agent_type.upper()
        print(f"  {symbol} ({agent_type}):")
        print(f"    Total Return: {result['total_return']:.2%}")
        print(f"    Final Portfolio: ${result['final_portfolio_value']:.2f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    agent_types = ["dqn", "sac", "dqn", "sac"]  # Mix of agent types
    
    # Train multi-agent system
    mas, summary = train_multi_agent_system(
        symbols=symbols,
        agent_types=agent_types,
        episodes=2000
    )
    
    print("\n" + "="*60)
    print("🎉 Multi-Agent Training Complete!")
    print("="*60)
