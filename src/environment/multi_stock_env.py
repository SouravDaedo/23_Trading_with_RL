import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import yaml

class MultiStockTradingEnvironment(gym.Env):
    """Multi-stock trading environment for simultaneous trading across multiple assets."""
    
    def __init__(self, stock_data: Dict[str, pd.DataFrame], config_path: str = "config/config.yaml"):
        """
        Initialize multi-stock trading environment.
        
        Args:
            stock_data: Dictionary with stock symbols as keys and DataFrames as values
                       e.g., {"AAPL": df_aapl, "GOOGL": df_googl, "MSFT": df_msft}
            config_path: Path to configuration file
        """
        super(MultiStockTradingEnvironment, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.stock_data = stock_data
        self.symbols = list(stock_data.keys())
        self.n_stocks = len(self.symbols)
        
        # Environment parameters
        self.lookback_window = self.config['data']['lookback_window']
        self.initial_balance = self.config['environment']['initial_balance']
        self.transaction_cost = self.config['environment']['transaction_cost']
        self.max_position = self.config['environment']['max_position']
        
        # Portfolio state
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}  # Position for each stock
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        # Action space: Multi-discrete (one action per stock)
        # Each stock can have 3 actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        
        # Observation space: Combined features from all stocks + portfolio state
        n_features_per_stock = len(next(iter(stock_data.values())).columns)
        total_market_features = self.lookback_window * n_features_per_stock * self.n_stocks
        portfolio_features = self.n_stocks + 2  # position per stock + balance + total_value
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_market_features + portfolio_features,),
            dtype=np.float32
        )
        
        # Align all stock data to same time index
        self._align_stock_data()
        
        # Extract price data for each stock
        self.prices = {}
        self._extract_price_data()
    
    def _align_stock_data(self):
        """Align all stock data to the same time index."""
        # Find common date range
        common_index = None
        for symbol, data in self.stock_data.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        # Align all data to common index
        for symbol in self.symbols:
            self.stock_data[symbol] = self.stock_data[symbol].loc[common_index]
        
        print(f"Aligned data: {len(common_index)} common time periods")
    
    def _extract_price_data(self):
        """Extract price data for each stock."""
        for symbol, data in self.stock_data.items():
            if 'Close' in data.columns:
                self.prices[symbol] = data['Close'].values
            else:
                # Fallback to first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.prices[symbol] = data[numeric_cols[0]].values
                else:
                    raise ValueError(f"No price data found for {symbol}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        return self._get_observation(), {}
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute actions for all stocks simultaneously.
        
        Args:
            actions: Array of actions, one per stock [action_stock1, action_stock2, ...]
        """
        if len(actions) != self.n_stocks:
            raise ValueError(f"Expected {self.n_stocks} actions, got {len(actions)}")
        
        prev_portfolio_value = self.portfolio_value
        total_reward = 0.0
        
        # Execute action for each stock
        for i, (symbol, action) in enumerate(zip(self.symbols, actions)):
            reward = self._execute_action(symbol, action)
            total_reward += reward
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(next(iter(self.stock_data.values()))) - 1
        truncated = False
        
        # Portfolio change reward
        portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        total_reward += portfolio_change * self.config['environment']['reward_scaling']
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'balance': self.balance,
            'step': self.current_step,
            'individual_rewards': {symbol: 0.0 for symbol in self.symbols}  # Could track per-stock rewards
        }
        
        return self._get_observation(), total_reward, done, truncated, info
    
    def _execute_action(self, symbol: str, action: int) -> float:
        """Execute trading action for a specific stock."""
        current_price = self.prices[symbol][self.current_step]
        reward = 0.0
        
        if action == 1:  # Buy
            if self.positions[symbol] < self.max_position:
                # Calculate how much we can buy (allocate balance across stocks)
                available_balance = self.balance / self.n_stocks  # Simple allocation
                max_shares = available_balance / current_price
                shares_to_buy = min(max_shares * 0.1, max_shares)  # Buy 10% or max available
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    if cost <= self.balance:
                        self.balance -= cost
                        self.positions[symbol] += shares_to_buy
                        reward = 0.01  # Small positive reward for successful trade
                        
                        self.trade_history.append({
                            'step': self.current_step,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
        
        elif action == 2:  # Sell
            if self.positions[symbol] > 0:
                shares_to_sell = self.positions[symbol]
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                
                self.balance += revenue
                self.positions[symbol] = 0.0
                reward = 0.01  # Small positive reward for successful trade
                
                self.trade_history.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue
                })
        
        # Action 0 (Hold) does nothing, reward = 0
        return reward
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        stock_value = 0.0
        for symbol, position in self.positions.items():
            current_price = self.prices[symbol][self.current_step]
            stock_value += position * current_price
        
        self.portfolio_value = self.balance + stock_value
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation combining all stocks and portfolio state."""
        # Market features for all stocks
        market_features = []
        
        for symbol in self.symbols:
            data = self.stock_data[symbol]
            
            # Get feature window for this stock
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step
            
            feature_window = data.iloc[start_idx:end_idx].values
            
            # Pad if necessary
            if len(feature_window) < self.lookback_window:
                padding = np.zeros((self.lookback_window - len(feature_window), feature_window.shape[1]))
                feature_window = np.vstack([padding, feature_window])
            
            # Flatten and add to market features
            market_features.extend(feature_window.flatten())
        
        # Portfolio state: positions for each stock + balance + total value
        portfolio_state = []
        for symbol in self.symbols:
            portfolio_state.append(self.positions[symbol])
        
        portfolio_state.extend([
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_state]).astype(np.float32)
        
        # Handle NaN/inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Per-stock positions
        stock_values = {}
        for symbol, position in self.positions.items():
            if self.current_step < len(self.prices[symbol]):
                current_price = self.prices[symbol][self.current_step]
                stock_values[symbol] = {
                    'position': position,
                    'current_price': current_price,
                    'market_value': position * current_price
                }
        
        return {
            'total_return': total_return,
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions,
            'stock_values': stock_values,
            'total_trades': len(self.trade_history),
            'current_step': self.current_step
        }
