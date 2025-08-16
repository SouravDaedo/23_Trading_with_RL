import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import time
from ..data.live_data_fetcher import LiveDataFetcher
from .trading_env import TradingEnvironment

class LiveTradingEnvironment(TradingEnvironment):
    """Live trading environment that uses real-time market data."""
    
    def __init__(self, symbol: str, config_path: str = "config/config.yaml", 
                 update_interval: int = 60):
        """
        Initialize live trading environment.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'TSLA')
            config_path: Path to configuration file
            update_interval: Data update interval in seconds
        """
        self.symbol = symbol
        self.update_interval = update_interval
        self.live_fetcher = LiveDataFetcher(symbol, config_path)
        
        # Initialize with historical data
        initial_data = self.live_fetcher.get_current_dataset()
        super().__init__(initial_data, config_path)
        
        self.live_mode = False
        self.last_update_time = time.time()
        
    def start_live_mode(self):
        """Start live trading mode with real-time data."""
        print(f"Starting live mode for {self.symbol}...")
        
        if not self.live_fetcher.is_market_open():
            print("⚠️  Market appears to be closed. Live data may be limited.")
        
        self.live_fetcher.start_streaming(self.update_interval)
        self.live_mode = True
        print("✅ Live mode activated!")
    
    def stop_live_mode(self):
        """Stop live trading mode."""
        self.live_fetcher.stop_streaming()
        self.live_mode = False
        print("Live mode stopped.")
    
    def _update_data_if_needed(self):
        """Update environment data with latest market data."""
        if not self.live_mode:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Get latest data
        latest_data = self.live_fetcher.get_latest_data()
        if latest_data is not None:
            # Update the environment's data
            self.data = self.live_fetcher.get_current_dataset()
            self.last_update_time = current_time
            
            print(f"📈 Data updated at {pd.Timestamp.now()}: "
                  f"Price=${latest_data.get('Close', 'N/A'):.2f}")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment with live data updates.
        """
        # Update data before taking action
        self._update_data_if_needed()
        
        # Execute the trading action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add live trading specific info
        info.update({
            'live_mode': self.live_mode,
            'market_open': self.live_fetcher.is_market_open(),
            'symbol': self.symbol,
            'timestamp': pd.Timestamp.now()
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment with latest data."""
        if self.live_mode:
            self._update_data_if_needed()
        
        return super().reset(**kwargs)
    
    def get_current_price(self) -> Optional[float]:
        """Get the current market price."""
        if self.current_step < len(self.data):
            return float(self.data.iloc[self.current_step]['Close'])
        return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and environment info."""
        current_price = self.get_current_price()
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'market_open': self.live_fetcher.is_market_open(),
            'live_mode': self.live_mode,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'current_step': self.current_step,
            'total_steps': len(self.data),
            'timestamp': pd.Timestamp.now()
        }
    
    def __del__(self):
        """Cleanup when environment is destroyed."""
        if hasattr(self, 'live_fetcher'):
            self.stop_live_mode()
