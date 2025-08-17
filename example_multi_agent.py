"""
Multi-Agent Trading System Example
Demonstrates how to use the multi-agent architecture for multi-stock trading.
"""

import yaml
from src.agents.multi_agent_system import MultiAgentTradingSystem
from src.data.data_loader import DataLoader

def run_multi_agent_example():
    """Run a simple multi-agent trading example."""
    print(" Multi-Agent Trading System Demo")
    print("=" * 50)
    
    # Configuration
    config_path = "config/config.yaml"
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    agent_types = ["dqn", "sac", "dqn", "sac"]  # Mix different agent types
    
    print(f"Configuration:")
    print(f"  Symbols: {symbols}")
    print(f"  Agent Types: {agent_types}")
    
    # Load data
    print(f"\n Loading market data...")
    data_loader = DataLoader(config_path)
    stock_data = {}
    
    for symbol in symbols:
        try:
            data = data_loader.load_data(symbol)
            if data is not None and not data.empty:
                stock_data[symbol] = data
                print(f"  {symbol}: {len(data)} data points")
            else:
                print(f"  {symbol}: No data available")
        except Exception as e:
            print(f"  {symbol}: Error loading data - {e}")
    
    if not stock_data:
        print(" No data loaded. Please check your data sources.")
        return
    
    # Update symbols and agent_types based on available data
    available_symbols = list(stock_data.keys())
    available_agent_types = agent_types[:len(available_symbols)]
    
    print(f"\n Initializing Multi-Agent System...")
    print(f"  Available symbols: {available_symbols}")
    
    # Initialize multi-agent system
    mas = MultiAgentTradingSystem(available_symbols, available_agent_types, config_path)
    mas.setup_agents(stock_data)
    
    print(f"\ System Architecture:")
    print(f"  Stock Agents: {len(mas.stock_agents)}")
    print(f"  Portfolio Allocation Agent: ")
    
    for symbol, agent in mas.stock_agents.items():
        agent_type = agent.agent_type.upper()
        env = mas.environments[symbol]
        state_size = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            action_size = env.action_space.n
        else:
            action_size = env.action_space.shape[0]
        print(f"    {symbol}: {agent_type} (State: {state_size}, Action: {action_size})")
    
    # Run a few training episodes as demonstration
    print(f"\ Running demonstration episodes...")
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Train one episode
        episode_results = mas.train_episode()
        
        # Show results
        total_reward = sum(sum(rewards) for rewards in episode_results['stock_rewards'].values())
        final_portfolio = episode_results['portfolio_values'][-1] if episode_results['portfolio_values'] else 0
        
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Final Portfolio: ${final_portfolio:.2f}")
        
        # Show individual stock performance
        for symbol in available_symbols:
            if symbol in episode_results['stock_rewards']:
                stock_reward = sum(episode_results['stock_rewards'][symbol])
                agent_type = mas.stock_agents[symbol].agent_type.upper()
                print(f"    {symbol} ({agent_type}): {stock_reward:.4f}")
        
        # Show allocation
        if episode_results['allocations']:
            final_allocation = episode_results['allocations'][-1]
            print(f"  Final Allocation:")
            for i, symbol in enumerate(available_symbols):
                print(f"    {symbol}: {final_allocation[i]:.2%}")
    
    # Show system statistics
    print(f"\n System Statistics:")
    stats = mas.get_system_stats()
    
    for symbol, agent_stats in stats['stock_agents'].items():
        agent_type = mas.stock_agents[symbol].agent_type.upper()
        print(f"  {symbol} ({agent_type}):")
        print(f"    Win Rate: {agent_stats['win_rate']:.2%}")
        print(f"    Avg Reward: {agent_stats['avg_reward']:.4f}")
        print(f"    Total Trades: {agent_stats['trade_count']}")
    
    portfolio_metrics = stats['portfolio_metrics']
    print(f"\n💼 Portfolio Metrics:")
    print(f"  Total Value: ${portfolio_metrics['total_value']:.2f}")
    print(f"  Total Return: {portfolio_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
    
    print(f"\n✅ Multi-Agent Demo Complete!")
    print(f"\nTo train the full system, run:")
    print(f"  python train_multi_agent.py")

if __name__ == "__main__":
    run_multi_agent_example()
