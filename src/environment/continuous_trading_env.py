import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from .trading_env import TradingEnvironment

class ContinuousTradingEnvironment(TradingEnvironment):
    """Continuous action space trading environment for SAC agent."""
    
    def __init__(self, data: pd.DataFrame, config_path: str = "config/config.yaml"):
        """Initialize continuous trading environment."""
        super().__init__(data, config_path)
        
        # Override action space for continuous control
        # Action: [position_change] where position_change is in [-1, 1]
        # -1 = sell all, 0 = hold, +1 = buy all
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        print(f"Continuous action space: {self.action_space}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute continuous action in the environment.
        
        Args:
            action: Continuous action in [-1, 1] range
                   -1.0 = maximum sell, 0.0 = hold, +1.0 = maximum buy
        """
        # Ensure action is in correct format
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        if action.ndim == 0:
            action = np.array([action])
        
        # Clamp action to valid range
        action = np.clip(action, -1.0, 1.0)[0]
        
        prev_portfolio_value = self.portfolio_value
        
        # Get current price before executing action
        current_price = self.prices[self.current_step]
        
        # Execute continuous action
        reward = self._execute_continuous_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate portfolio value change
        portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Additional reward shaping
        reward += portfolio_change * self.reward_scaling
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'step': self.current_step,
            'action_value': float(action),
            'current_price': current_price
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_continuous_action(self, action: float) -> float:
        """
        Execute continuous trading action.
        
        Args:
            action: Continuous action value in [-1, 1]
                   -1.0 = sell all holdings
                    0.0 = hold current position
                   +1.0 = buy maximum possible
        """
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        # Calculate target position based on action
        if action > 0:
            # Buying action: action = 1.0 means use all available balance
            available_balance = self.balance
            max_shares = available_balance / current_price
            target_shares = max_shares * action * 0.95  # Use 95% to account for transaction costs
            
            if target_shares > 0:
                cost = target_shares * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    current_position_value = self.position * current_price
                    self.position += target_shares
                    reward = 0.01 * action  # Small positive reward proportional to action
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': target_shares,
                        'price': current_price,
                        'cost': cost,
                        'action_value': action
                    })
        
        elif action < 0:
            # Selling action: action = -1.0 means sell all holdings
            if self.position > 0:
                shares_to_sell = self.position * abs(action)  # Sell proportion based on action magnitude
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                
                self.balance += revenue
                self.position -= shares_to_sell
                self.position = max(0, self.position)  # Ensure position doesn't go negative
                reward = 0.01 * abs(action)  # Small positive reward proportional to action
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue,
                    'action_value': action
                })
        
        # Action close to 0 (hold) gets no reward/penalty
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.position * current_price)
        
        return reward
    
    def get_action_interpretation(self, action: float) -> str:
        """Get human-readable interpretation of continuous action."""
        action = float(action)
        
        if action > 0.8:
            return f"STRONG BUY ({action:.3f})"
        elif action > 0.3:
            return f"BUY ({action:.3f})"
        elif action > 0.1:
            return f"WEAK BUY ({action:.3f})"
        elif action > -0.1:
            return f"HOLD ({action:.3f})"
        elif action > -0.3:
            return f"WEAK SELL ({action:.3f})"
        elif action > -0.8:
            return f"SELL ({action:.3f})"
        else:
            return f"STRONG SELL ({action:.3f})"


class MultiActionContinuousEnvironment(ContinuousTradingEnvironment):
    """
    Continuous environment with multiple action dimensions.
    Actions: [position_change, risk_level]
    """
    
    def __init__(self, data: pd.DataFrame, config_path: str = "config/config.yaml"):
        """Initialize multi-action continuous environment."""
        super().__init__(data, config_path)
        
        # Override action space for multi-dimensional continuous control
        # Action: [position_change, risk_level]
        # position_change: [-1, 1] - trading decision
        # risk_level: [0, 1] - position sizing factor
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        print(f"Multi-action continuous space: {self.action_space}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute multi-dimensional continuous action."""
        # Ensure action has correct shape
        if len(action) != 2:
            raise ValueError(f"Expected 2D action, got {len(action)}D")
        
        position_action = np.clip(action[0], -1.0, 1.0)
        risk_level = np.clip(action[1], 0.0, 1.0)
        
        prev_portfolio_value = self.portfolio_value
        
        # Execute action with risk adjustment
        reward = self._execute_multi_action(position_action, risk_level)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate portfolio value change
        portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_change * self.reward_scaling
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'step': self.current_step,
            'position_action': float(position_action),
            'risk_level': float(risk_level)
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_multi_action(self, position_action: float, risk_level: float) -> float:
        """Execute multi-dimensional continuous action."""
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        # Apply risk level to position sizing
        effective_action = position_action * risk_level
        
        if effective_action > 0:
            # Buying with risk adjustment
            available_balance = self.balance
            max_shares = available_balance / current_price
            target_shares = max_shares * effective_action * 0.95
            
            if target_shares > 0:
                cost = target_shares * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += target_shares
                    reward = 0.01 * effective_action
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': target_shares,
                        'price': current_price,
                        'cost': cost,
                        'position_action': position_action,
                        'risk_level': risk_level,
                        'effective_action': effective_action
                    })
        
        elif effective_action < 0:
            # Selling with risk adjustment
            if self.position > 0:
                shares_to_sell = self.position * abs(effective_action)
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                
                self.balance += revenue
                self.position -= shares_to_sell
                self.position = max(0, self.position)
                reward = 0.01 * abs(effective_action)
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue,
                    'position_action': position_action,
                    'risk_level': risk_level,
                    'effective_action': effective_action
                })
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.position * current_price)
        
        return reward
