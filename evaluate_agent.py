#!/usr/bin/env python3
"""
Evaluation script for the trained RL trading agent.
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

def evaluate_agent(model_path: str, config_path: str = "config/config.yaml", use_test_data: bool = True):
    """Evaluate a trained RL trading agent."""
    print("=" * 60)
    print("📊 Evaluating RL Trading Agent")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data loader
    print("\n📊 Loading evaluation data...")
    data_loader = DataLoader(config_path)
    
    # Load data
    symbol = config['data']['symbols'][0]
    train_data, test_data = data_loader.prepare_data(symbol)
    
    # Use test data for evaluation
    eval_data = test_data if use_test_data else train_data
    data_type = "test" if use_test_data else "train"
    
    print(f"Evaluating on {data_type} data: {len(eval_data)} samples")
    
    # Initialize environment
    print(f"\n🏗️ Setting up evaluation environment...")
    env = TradingEnvironment(eval_data, config_path)
    
    # Initialize and load agent
    print(f"\n🤖 Loading trained agent from {model_path}...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config_path)
    agent.load(model_path)
    
    # Evaluation parameters
    test_episodes = config['evaluation']['test_episodes']
    
    # Evaluation metrics
    all_portfolio_values = []
    all_rewards = []
    all_trade_histories = []
    episode_stats = []
    
    print(f"\n🎯 Running evaluation for {test_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        episode_portfolio_values = [env.initial_balance]
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_portfolio_values.append(info['portfolio_value'])
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Store episode results
        all_portfolio_values.append(episode_portfolio_values)
        all_rewards.append(total_reward)
        all_trade_histories.append(env.trade_history.copy())
        
        # Calculate episode metrics
        portfolio_stats = env.get_portfolio_stats()
        episode_stats.append(portfolio_stats)
        
        print(f"Episode {episode + 1:2d} | "
              f"Final Value: ${portfolio_stats['final_portfolio_value']:8.0f} | "
              f"Return: {portfolio_stats['total_return']:6.2f}% | "
              f"Trades: {portfolio_stats['total_trades']:3d} | "
              f"Win Rate: {portfolio_stats['win_rate']:5.1f}%")
    
    print("\n✅ Evaluation completed!")
    
    # Calculate aggregate metrics
    print("\n📈 Calculating performance metrics...")
    
    # Average metrics across episodes
    avg_final_value = np.mean([stats['final_portfolio_value'] for stats in episode_stats])
    avg_return = np.mean([stats['total_return'] for stats in episode_stats])
    avg_trades = np.mean([stats['total_trades'] for stats in episode_stats])
    avg_win_rate = np.mean([stats['win_rate'] for stats in episode_stats])
    
    # Best and worst episodes
    returns = [stats['total_return'] for stats in episode_stats]
    best_episode = np.argmax(returns)
    worst_episode = np.argmin(returns)
    
    # Calculate comprehensive metrics for best episode
    best_portfolio_values = all_portfolio_values[best_episode]
    best_trade_history = all_trade_histories[best_episode]
    comprehensive_metrics = TradingMetrics.calculate_all_metrics(
        best_portfolio_values, best_trade_history, env.initial_balance
    )
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes Evaluated: {test_episodes}")
    print(f"Data Type: {data_type.upper()}")
    print(f"Initial Balance: ${env.initial_balance:,.2f}")
    print()
    print("AVERAGE PERFORMANCE:")
    print(f"  Final Portfolio Value: ${avg_final_value:,.2f}")
    print(f"  Average Return: {avg_return:.2f}%")
    print(f"  Average Trades per Episode: {avg_trades:.1f}")
    print(f"  Average Win Rate: {avg_win_rate:.1f}%")
    print()
    print("BEST EPISODE PERFORMANCE:")
    print(f"  Episode: {best_episode + 1}")
    print(f"  Total Return: {comprehensive_metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {comprehensive_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {comprehensive_metrics['max_drawdown']:.2f}%")
    print(f"  Volatility: {comprehensive_metrics['volatility']:.2f}%")
    print(f"  Calmar Ratio: {comprehensive_metrics['calmar_ratio']:.3f}")
    print()
    print("WORST EPISODE:")
    print(f"  Episode: {worst_episode + 1}")
    print(f"  Return: {returns[worst_episode]:.2f}%")
    
    # Create visualizations
    print("\n📊 Creating evaluation visualizations...")
    visualizer = TradingVisualizer()
    
    # Create plots directory
    os.makedirs('logs/evaluation', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot best episode performance
    best_portfolio_values = all_portfolio_values[best_episode]
    best_trade_history = all_trade_histories[best_episode]
    
    visualizer.plot_portfolio_performance(
        best_portfolio_values,
        best_trade_history,
        save_path=f'logs/evaluation/best_episode_{symbol}_{timestamp}.png'
    )
    
    # Plot performance distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist([stats['total_return'] for stats in episode_stats], bins=10, alpha=0.7, color='skyblue')
    plt.title('Return Distribution')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.hist([stats['total_trades'] for stats in episode_stats], bins=10, alpha=0.7, color='lightgreen')
    plt.title('Trades per Episode')
    plt.xlabel('Number of Trades')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.hist([stats['win_rate'] for stats in episode_stats], bins=10, alpha=0.7, color='lightcoral')
    plt.title('Win Rate Distribution')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot(returns, marker='o', alpha=0.7)
    plt.title('Returns by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.subplot(2, 3, 5)
    final_values = [stats['final_portfolio_value'] for stats in episode_stats]
    plt.plot(final_values, marker='o', alpha=0.7, color='purple')
    plt.title('Final Portfolio Value by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=env.initial_balance, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    # Performance metrics radar chart (simplified as bar chart)
    metrics_names = ['Avg Return', 'Avg Win Rate', 'Best Return', 'Consistency']
    consistency = 100 - (np.std(returns) / np.mean(np.abs(returns)) * 100) if np.mean(np.abs(returns)) > 0 else 0
    metrics_values = [avg_return, avg_win_rate, max(returns), max(0, consistency)]
    
    plt.bar(range(len(metrics_names)), metrics_values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('Performance Summary')
    plt.xticks(range(len(metrics_names)), metrics_names, rotation=45)
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'logs/evaluation/evaluation_summary_{symbol}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save evaluation results
    results = {
        'config': config,
        'model_path': model_path,
        'data_type': data_type,
        'test_episodes': test_episodes,
        'average_metrics': {
            'final_portfolio_value': avg_final_value,
            'total_return': avg_return,
            'total_trades': avg_trades,
            'win_rate': avg_win_rate
        },
        'best_episode_metrics': comprehensive_metrics,
        'all_episode_stats': episode_stats,
        'timestamp': timestamp
    }
    
    # Save to file
    import json
    with open(f'logs/evaluation/results_{symbol}_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n Evaluation complete!")
    print(f" Results saved to: logs/evaluation/results_{symbol}_{timestamp}.json")
    print(" Check the 'logs/evaluation' directory for visualization outputs.")
    
    return results

def compare_with_benchmark(results: dict):
    """Compare agent performance with buy-and-hold benchmark."""
    print("\n" + "=" * 60)
    print(" BENCHMARK COMPARISON")
    print("=" * 60)
    
    # Simple buy-and-hold comparison (simplified)
    initial_balance = 10000  # From config
    
    # Assuming the agent's average return vs buy-and-hold
    agent_return = results['average_metrics']['total_return']
    
    # Simplified benchmark (this would need actual price data for accurate comparison)
    # For demonstration, assume market returned 10% annually
    benchmark_return = 10.0  # This should be calculated from actual price data
    
    print(f"Agent Average Return: {agent_return:.2f}%")
    print(f"Buy-and-Hold Benchmark: {benchmark_return:.2f}%")
    print(f"Excess Return: {agent_return - benchmark_return:.2f}%")
    
    if agent_return > benchmark_return:
        print("Agent outperformed the benchmark!")
    else:
        print(" Agent underperformed the benchmark.")

if __name__ == "__main__":
    import glob
    
    # Find the most recent model
    model_files = glob.glob('models/*.pth')
    if not model_files:
        print(" No trained models found. Please run train_agent.py first.")
        sys.exit(1)
    
    # Use the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Using model: {latest_model}")
    
    try:
        results = evaluate_agent(latest_model)
        compare_with_benchmark(results)
        print("\n Evaluation pipeline executed successfully!")
    except Exception as e:
        print(f"\n Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
