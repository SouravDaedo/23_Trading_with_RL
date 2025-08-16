import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import yaml

class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning."""
    
    def __init__(self, data: pd.DataFrame, config_path: str = "config/config.yaml"):
        """Initialize the trading environment."""
        super(TradingEnvironment, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data = data.copy()
        self.lookback_window = self.config['data']['lookback_window']
        self.initial_balance = self.config['environment']['initial_balance']
        self.transaction_cost = self.config['environment']['transaction_cost']
        self.max_position = self.config['environment']['max_position']
        self.reward_scaling = self.config['environment']['reward_scaling']
        
        # Environment state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + portfolio state
        n_features = len(self.data.columns)
        portfolio_features = 4  # balance, position, portfolio_value, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback_window * n_features + portfolio_features,),
            dtype=np.float32
        )
        
        # Price data for calculating returns
        self.prices = None
        self._extract_price_data()
        
    def _extract_price_data(self):
        """Extract price data for portfolio calculations."""
        # Try to find price column
        price_cols = [col for col in self.data.columns if 'Close' in col or 'price' in col.lower()]
        if price_cols:
            # Use normalized close price and denormalize it
            if 'Close_norm' in self.data.columns:
                # Approximate denormalization (this is simplified)
                self.prices = self.data['Close_norm'].values
            else:
                self.prices = self.data[price_cols[0]].values
        else:
            # Fallback: use first column as price
            self.prices = self.data.iloc[:, 0].values
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate portfolio value change
        portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Additional reward shaping
        reward += portfolio_change * self.reward_scaling
        
        # Penalty for excessive trading
        if action != 0:  # If not holding
            reward -= self.transaction_cost
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the trading action and return immediate reward."""
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        if action == 1:  # Buy
            if self.position < self.max_position:
                # Calculate how much we can buy
                max_shares = self.balance / current_price
                shares_to_buy = min(max_shares * 0.1, max_shares)  # Buy 10% or max available
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    if cost <= self.balance:
                        self.balance -= cost
                        self.position += shares_to_buy / (self.initial_balance / current_price)  # Normalize position
                        self.trade_history.append(('BUY', shares_to_buy, current_price, self.current_step))
                        reward = 0.01  # Small positive reward for taking action
        
        elif action == 2:  # Sell
            if self.position > -self.max_position:
                # Calculate how much we can sell
                shares_to_sell = abs(self.position) * (self.initial_balance / current_price) * 0.1  # Sell 10%
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                    self.balance += proceeds
                    self.position -= shares_to_sell / (self.initial_balance / current_price)  # Normalize position
                    self.trade_history.append(('SELL', shares_to_sell, current_price, self.current_step))
                    reward = 0.01  # Small positive reward for taking action
        
        # Action 0 (hold) requires no execution
        
        # Update portfolio value
        position_value = self.position * (self.initial_balance / current_price) * current_price
        self.portfolio_value = self.balance + position_value
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Get feature window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        feature_window = self.data.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if len(feature_window) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(feature_window), feature_window.shape[1]))
            feature_window = np.vstack([padding, feature_window])
        
        # Flatten feature window
        features = feature_window.flatten()
        
        # Add portfolio state
        current_price = self.prices[self.current_step] if self.current_step < len(self.prices) else self.prices[-1]
        unrealized_pnl = self.position * (self.initial_balance / current_price) * current_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            unrealized_pnl / self.initial_balance  # Normalized unrealized P&L
        ])
        
        # Combine features and portfolio state
        observation = np.concatenate([features, portfolio_state]).astype(np.float32)
        
        # Handle any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def get_portfolio_stats(self) -> Dict:
        """Calculate portfolio performance statistics."""
        if len(self.trade_history) == 0:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'final_portfolio_value': self.portfolio_value
            }
        
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        total_trades = len(self.trade_history)
        
        # Calculate win rate (simplified)
        profitable_trades = 0
        for i in range(1, len(self.trade_history)):
            if self.trade_history[i][0] == 'SELL' and i > 0:
                buy_price = self.trade_history[i-1][2] if self.trade_history[i-1][0] == 'BUY' else 0
                sell_price = self.trade_history[i][2]
                if sell_price > buy_price:
                    profitable_trades += 1
        
        win_rate = profitable_trades / max(1, total_trades // 2)  # Approximate win rate
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'trade_history': self.trade_history
        }
