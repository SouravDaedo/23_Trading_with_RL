#!/usr/bin/env python3
"""
Live Testing Script for RL Trading Agent
Test your trained agent with real-time market data.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.live_trading_env import LiveTradingEnvironment
from src.agents.dqn_agent import DQNAgent

def load_trained_agent(model_path: str, env: LiveTradingEnvironment) -> DQNAgent:
    """Load a pre-trained DQN agent."""
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("Using untrained agent (random actions)")
    
    return agent

def print_status(env: LiveTradingEnvironment, action: int, reward: float, step: int):
    """Print current trading status."""
    status = env.get_market_status()
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    print(f"\n{'='*60}")
    print(f"Step {step} | {status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: {status['symbol']} | Price: ${status['current_price']:.2f}")
    print(f"Action: {action_names[action]} | Reward: {reward:.4f}")
    print(f"Portfolio Value: ${status['portfolio_value']:.2f}")
    print(f"Position: {status['position']:.3f} | Balance: ${status['balance']:.2f}")
    print(f"Market Open: {'✅' if status['market_open'] else '❌'}")
    print(f"{'='*60}")

def live_test_agent(symbol: str, model_path: str, max_steps: int = 100, 
                   update_interval: int = 60):
    """
    Test the trained agent with live market data.
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'TSLA')
        model_path: Path to trained model
        max_steps: Maximum number of trading steps
        update_interval: Data update interval in seconds
    """
    print(f"🚀 Starting live test for {symbol}")
    print(f"Update interval: {update_interval} seconds")
    print(f"Max steps: {max_steps}")
    
    # Initialize environment
    env = LiveTradingEnvironment(
        symbol=symbol, 
        update_interval=update_interval
    )
    
    # Load trained agent
    agent = load_trained_agent(model_path, env)
    
    try:
        # Start live mode
        env.start_live_mode()
        
        # Reset environment
        state, info = env.reset()
        total_reward = 0
        step = 0
        
        print(f"\n🎯 Initial Portfolio Value: ${env.portfolio_value:.2f}")
        
        while step < max_steps:
            # Get action from agent
            action = agent.act(state, epsilon=0)  # No exploration in testing
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update total reward
            total_reward += reward
            
            # Print status
            print_status(env, action, reward, step)
            
            # Check if episode ended
            if terminated or truncated:
                print(f"\n🏁 Episode ended at step {step}")
                break
            
            # Update state
            state = next_state
            step += 1
            
            # Wait for next update (if not in fast mode)
            if update_interval > 10:
                print(f"⏳ Waiting {update_interval} seconds for next update...")
                time.sleep(update_interval)
        
        # Final results
        final_value = env.portfolio_value
        initial_value = env.initial_balance
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"\n📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Initial Portfolio Value: ${initial_value:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Steps Completed: {step}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Live test interrupted by user")
    
    finally:
        # Stop live mode
        env.stop_live_mode()
        print("Live test completed.")

def paper_trading_mode(symbol: str, model_path: str, duration_hours: int = 1):
    """
    Run continuous paper trading for a specified duration.
    
    Args:
        symbol: Trading symbol
        model_path: Path to trained model
        duration_hours: How long to run (in hours)
    """
    print(f"📄 Starting paper trading mode for {symbol}")
    print(f"Duration: {duration_hours} hours")
    
    env = LiveTradingEnvironment(symbol=symbol, update_interval=300)  # 5 min updates
    agent = load_trained_agent(model_path, env)
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    try:
        env.start_live_mode()
        state, _ = env.reset()
        
        trades_made = 0
        
        while time.time() < end_time:
            action = agent.act(state, epsilon=0)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if action != 0:  # Not holding
                trades_made += 1
                print(f"🔄 Trade #{trades_made}: {['HOLD', 'BUY', 'SELL'][action]} at ${info.get('current_price', 'N/A')}")
            
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
            
            time.sleep(300)  # Wait 5 minutes
        
        print(f"\n📄 Paper trading completed. Total trades: {trades_made}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Paper trading interrupted")
    finally:
        env.stop_live_mode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live test RL trading agent")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--model", default="models/dqn_model.pth", help="Path to trained model")
    parser.add_argument("--steps", type=int, default=50, help="Maximum steps")
    parser.add_argument("--interval", type=int, default=60, help="Update interval (seconds)")
    parser.add_argument("--mode", choices=["test", "paper"], default="test", help="Testing mode")
    parser.add_argument("--duration", type=int, default=1, help="Duration for paper trading (hours)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        live_test_agent(
            symbol=args.symbol,
            model_path=args.model,
            max_steps=args.steps,
            update_interval=args.interval
        )
    elif args.mode == "paper":
        paper_trading_mode(
            symbol=args.symbol,
            model_path=args.model,
            duration_hours=args.duration
        )
