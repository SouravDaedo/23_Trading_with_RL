"""
Multi-Agent Trading System
Coordinates multiple specialized agents for multi-stock trading.
"""

import numpy as np
import pandas as pd
import torch
import yaml
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import os

from .dqn_agent import DQNAgent
from .sac_agent import SACAgent
from ..environment.trading_env import TradingEnvironment
from ..environment.continuous_trading_env import ContinuousTradingEnvironment

class StockAgent:
    """Individual agent specialized for a specific stock."""
    
    def __init__(self, symbol: str, agent_type: str, state_size: int, action_size: int, 
                 config_path: str = "config/config.yaml"):
        """
        Initialize stock-specific agent.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            agent_type: 'dqn' or 'sac'
            state_size: Size of observation space
            action_size: Size of action space
            config_path: Path to configuration file
        """
        self.symbol = symbol
        self.agent_type = agent_type
        self.state_size = state_size
        self.action_size = action_size
        
        # Create specialized agent
        if agent_type.lower() == 'dqn':
            self.agent = DQNAgent(state_size, action_size, config_path)
        elif agent_type.lower() == 'sac':
            self.agent = SACAgent(state_size, action_size, config_path)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Performance tracking
        self.performance_history = []
        self.trade_count = 0
        self.successful_trades = 0
        self.total_return = 0.0
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Any:
        """Select action using the specialized agent."""
        if self.agent_type == 'dqn':
            epsilon = 0.0 if evaluate else self.agent.epsilon
            return self.agent.act(state, epsilon)
        else:  # SAC
            return self.agent.select_action(state, evaluate)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in agent's memory."""
        if self.agent_type == 'dqn':
            self.agent.remember(state, action, reward, next_state, done)
        else:  # SAC
            self.agent.store_transition(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update agent parameters."""
        if self.agent_type == 'dqn':
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()
                return {'loss': loss}
            return {'loss': 0.0}
        else:  # SAC
            if len(self.agent.memory) > self.agent.batch_size:
                return self.agent.update_parameters()
            return {}
    
    def update_performance(self, reward: float, portfolio_value: float):
        """Update performance metrics."""
        self.performance_history.append({
            'reward': reward,
            'portfolio_value': portfolio_value,
            'timestamp': pd.Timestamp.now()
        })
        
        if reward > 0:
            self.successful_trades += 1
        self.trade_count += 1
        self.total_return += reward
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_history:
            return {'win_rate': 0.0, 'avg_reward': 0.0, 'total_return': 0.0}
        
        win_rate = self.successful_trades / max(1, self.trade_count)
        avg_reward = np.mean([p['reward'] for p in self.performance_history])
        
        return {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'total_return': self.total_return,
            'trade_count': self.trade_count
        }
    
    def save(self, filepath: str):
        """Save the specialized agent."""
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """Load the specialized agent."""
        self.agent.load(filepath)


class PortfolioAllocationAgent:
    """Meta-agent that decides capital allocation across stock agents."""
    
    def __init__(self, n_stocks: int, config_path: str = "config/config.yaml"):
        """
        Initialize portfolio allocation agent.
        
        Args:
            n_stocks: Number of stocks in portfolio
            config_path: Path to configuration file
        """
        self.n_stocks = n_stocks
        
        # State: [stock_performances, portfolio_metrics, market_conditions]
        # Action: allocation weights for each stock [0, 1] that sum to 1
        state_size = n_stocks * 3 + 5  # 3 metrics per stock + 5 portfolio metrics
        action_size = n_stocks  # Allocation weight for each stock
        
        # Use SAC for continuous allocation decisions
        self.agent = SACAgent(state_size, action_size, config_path)
        
        self.allocation_history = []
        self.performance_history = []
    
    def get_allocation_state(self, stock_agents: List[StockAgent], 
                           portfolio_metrics: Dict[str, float]) -> np.ndarray:
        """Create state representation for allocation decisions."""
        state_components = []
        
        # Stock agent performances
        for agent in stock_agents:
            stats = agent.get_performance_stats()
            state_components.extend([
                stats['win_rate'],
                stats['avg_reward'],
                stats['total_return']
            ])
        
        # Portfolio-level metrics
        state_components.extend([
            portfolio_metrics.get('total_value', 0.0),
            portfolio_metrics.get('total_return', 0.0),
            portfolio_metrics.get('volatility', 0.0),
            portfolio_metrics.get('sharpe_ratio', 0.0),
            portfolio_metrics.get('max_drawdown', 0.0)
        ])
        
        return np.array(state_components, dtype=np.float32)
    
    def select_allocation(self, stock_agents: List[StockAgent], 
                         portfolio_metrics: Dict[str, float], 
                         evaluate: bool = False) -> np.ndarray:
        """Select capital allocation across stocks."""
        state = self.get_allocation_state(stock_agents, portfolio_metrics)
        raw_allocation = self.agent.select_action(state, evaluate)
        
        # Convert to valid allocation (softmax to ensure sum = 1)
        allocation = np.exp(raw_allocation) / np.sum(np.exp(raw_allocation))
        allocation = np.clip(allocation, 0.01, 0.99)  # Prevent extreme allocations
        allocation = allocation / np.sum(allocation)  # Renormalize
        
        self.allocation_history.append({
            'allocation': allocation.copy(),
            'timestamp': pd.Timestamp.now()
        })
        
        return allocation
    
    def update(self, state, action, reward, next_state, done):
        """Update allocation agent."""
        self.agent.store_transition(state, action, reward, next_state, done)
        
        if len(self.agent.memory) > self.agent.batch_size:
            return self.agent.update_parameters()
        return {}


class MultiAgentTradingSystem:
    """Coordinates multiple agents for multi-stock trading."""
    
    def __init__(self, symbols: List[str], agent_types: List[str], 
                 config_path: str = "config/config.yaml"):
        """
        Initialize multi-agent trading system.
        
        Args:
            symbols: List of stock symbols
            agent_types: List of agent types for each stock ('dqn' or 'sac')
            config_path: Path to configuration file
        """
        self.symbols = symbols
        self.agent_types = agent_types
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize stock agents
        self.stock_agents = {}
        self.environments = {}
        
        # Portfolio allocation agent
        self.allocation_agent = PortfolioAllocationAgent(len(symbols), config_path)
        
        # Portfolio tracking
        self.initial_balance = self.config['environment']['initial_balance']
        self.current_balance = self.initial_balance
        self.portfolio_history = []
        
        print(f"🤖 Multi-Agent System initialized:")
        print(f"  Stocks: {symbols}")
        print(f"  Agent types: {agent_types}")
        print(f"  Initial balance: ${self.initial_balance:,.2f}")
    
    def setup_agents(self, stock_data: Dict[str, pd.DataFrame]):
        """Setup individual stock agents and environments."""
        for i, symbol in enumerate(self.symbols):
            agent_type = self.agent_types[i]
            data = stock_data[symbol]
            
            # Create environment
            if agent_type == 'dqn':
                env = TradingEnvironment(data, self.config_path)
                action_size = env.action_space.n
            else:  # SAC
                env = ContinuousTradingEnvironment(data, self.config_path)
                action_size = env.action_space.shape[0]
            
            self.environments[symbol] = env
            
            # Create specialized agent
            state_size = env.observation_space.shape[0]
            self.stock_agents[symbol] = StockAgent(
                symbol, agent_type, state_size, action_size, self.config_path
            )
            
            print(f"  ✅ {symbol}: {agent_type.upper()} agent (state: {state_size}, action: {action_size})")
    
    def train_episode(self) -> Dict[str, Any]:
        """Train all agents for one episode."""
        episode_results = {
            'stock_rewards': defaultdict(list),
            'stock_actions': defaultdict(list),
            'allocations': [],
            'portfolio_values': []
        }
        
        # Reset all environments
        states = {}
        for symbol in self.symbols:
            state, _ = self.environments[symbol].reset()
            states[symbol] = state
        
        # Episode loop
        done = False
        step = 0
        
        while not done:
            # Get current portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics()
            
            # Get allocation from meta-agent
            allocation = self.allocation_agent.select_allocation(
                list(self.stock_agents.values()), portfolio_metrics
            )
            episode_results['allocations'].append(allocation.copy())
            
            # Execute actions for each stock
            stock_rewards = {}
            next_states = {}
            
            for i, symbol in enumerate(self.symbols):
                # Get action from stock agent
                agent = self.stock_agents[symbol]
                action = agent.select_action(states[symbol])
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = \
                    self.environments[symbol].step(action)
                
                # Scale reward by allocation
                scaled_reward = reward * allocation[i]
                stock_rewards[symbol] = scaled_reward
                
                # Store experience
                agent.store_experience(
                    states[symbol], action, scaled_reward, next_state, 
                    terminated or truncated
                )
                
                # Update agent
                update_info = agent.update()
                
                # Update performance
                agent.update_performance(scaled_reward, info['portfolio_value'])
                
                # Store results
                episode_results['stock_rewards'][symbol].append(scaled_reward)
                episode_results['stock_actions'][symbol].append(action)
                
                next_states[symbol] = next_state
                
                # Check if any environment is done
                if terminated or truncated:
                    done = True
            
            # Update allocation agent
            total_reward = sum(stock_rewards.values())
            # (Allocation agent update would go here with proper state transitions)
            
            # Update states
            states = next_states
            step += 1
            
            # Record portfolio value
            total_portfolio_value = sum(
                self.environments[symbol].portfolio_value 
                for symbol in self.symbols
            )
            episode_results['portfolio_values'].append(total_portfolio_value)
        
        return episode_results
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level performance metrics."""
        total_value = sum(
            env.portfolio_value for env in self.environments.values()
        )
        
        total_return = (total_value - self.initial_balance) / self.initial_balance
        
        # Calculate volatility from recent portfolio values
        if len(self.portfolio_history) > 10:
            recent_values = [p['total_value'] for p in self.portfolio_history[-10:]]
            returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.0
        
        return {
            'total_value': total_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': total_return / max(volatility, 0.001),
            'max_drawdown': 0.0  # Would need proper calculation
        }
    
    def evaluate_agents(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate all agents on test data."""
        results = {}
        
        for symbol in self.symbols:
            # Setup test environment
            agent_type = self.stock_agents[symbol].agent_type
            if agent_type == 'dqn':
                test_env = TradingEnvironment(test_data[symbol], self.config_path)
            else:
                test_env = ContinuousTradingEnvironment(test_data[symbol], self.config_path)
            
            # Run evaluation
            state, _ = test_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.stock_agents[symbol].select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = test_env.step(action)
                
                total_reward += reward
                state = next_state
                done = terminated or truncated
            
            results[symbol] = {
                'total_reward': total_reward,
                'final_portfolio_value': test_env.portfolio_value,
                'total_return': (test_env.portfolio_value - self.initial_balance) / self.initial_balance
            }
        
        return results
    
    def save_all_agents(self, base_path: str = "models/multi_agent"):
        """Save all agents."""
        os.makedirs(base_path, exist_ok=True)
        
        for symbol, agent in self.stock_agents.items():
            agent_path = os.path.join(base_path, f"{symbol}_{agent.agent_type}.pth")
            agent.save(agent_path)
        
        # Save allocation agent
        allocation_path = os.path.join(base_path, "allocation_agent.pth")
        self.allocation_agent.agent.save(allocation_path)
        
        print(f"💾 All agents saved to {base_path}")
    
    def load_all_agents(self, base_path: str = "models/multi_agent"):
        """Load all agents."""
        for symbol, agent in self.stock_agents.items():
            agent_path = os.path.join(base_path, f"{symbol}_{agent.agent_type}.pth")
            if os.path.exists(agent_path):
                agent.load(agent_path)
        
        # Load allocation agent
        allocation_path = os.path.join(base_path, "allocation_agent.pth")
        if os.path.exists(allocation_path):
            self.allocation_agent.agent.load(allocation_path)
        
        print(f"📁 All agents loaded from {base_path}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'stock_agents': {},
            'allocation_history': self.allocation_agent.allocation_history[-10:],
            'portfolio_metrics': self._calculate_portfolio_metrics()
        }
        
        for symbol, agent in self.stock_agents.items():
            stats['stock_agents'][symbol] = agent.get_performance_stats()
        
        return stats
