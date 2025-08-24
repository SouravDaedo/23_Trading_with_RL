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
                 update_interval: int = None):
        """
        Initialize live trading environment.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'TSLA')
            config_path: Path to configuration file
            update_interval: Data update interval in seconds. If None, will be set based on timeframe.
        """
        self.symbol = symbol
        self.config_path = config_path
        
        # Initialize data fetcher first to get timeframe info
        self.live_fetcher = LiveDataFetcher(symbol, config_path)
        
        # Set update interval based on timeframe if not specified
        if update_interval is None:
            # Default update intervals based on timeframe
            timeframe_intervals = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400, '1wk': 604800, '1mo': 2592000
            }
            self.update_interval = timeframe_intervals.get(
                self.live_fetcher.data_loader.timeframe, 3600
            )
        else:
            self.update_interval = update_interval
        
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
        time_since_last_update = current_time - self.last_update_time
        
        # Check if it's time to update based on the timeframe
        if time_since_last_update < self.update_interval:
            return
        
        try:
            # Get latest data
            latest_data = self.live_fetcher.get_latest_data()
            if latest_data is not None and not latest_data.empty:
                # Get the complete updated dataset
                updated_data = self.live_fetcher.get_current_dataset()
                
                # Only update if we have new data
                if len(updated_data) > len(self.data) or \
                   (len(updated_data) > 0 and len(self.data) > 0 and 
                    updated_data.index[-1] > self.data.index[-1]):
                    
                    # Store the current position and balance before updating
                    prev_position = self.position
                    prev_balance = self.balance
                    
                    # Update the data
                    self.data = updated_data
                    self.last_update_time = current_time
                    
                    # Reinitialize price data after update
                    self._extract_price_data()
                    
                    # Log the update
                    current_price = latest_data.get('Close', float('nan'))
                    print(f"📈 {self.symbol} {self.timeframe} - "
                          f"Price: ${current_price:.2f} | "
                          f"Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                          f"Candles: {len(self.data)}")
                    
                    # If we're in the middle of an episode, adjust the current step
                    if hasattr(self, 'current_step') and self.current_step < len(self.data):
                        # Update the current step to the latest data point
                        self.current_step = len(self.data) - 1
                        
                        # Restore position and balance
                        self.position = prev_position
                        self.balance = prev_balance
                        
                        # Recalculate portfolio value
                        self.portfolio_value = self.balance + (
                            self.position * self.prices[self.current_step]
                        )
        except Exception as e:
            print(f"⚠️  Error updating market data: {str(e)}")
            # If there's an error, try to continue with existing data
            self.last_update_time = current_time - (self.update_interval // 2)  # Retry sooner
    
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
        
        # Calculate position value if we have a current price
        position_value = 0.0
        if current_price is not None and hasattr(self, 'position'):
            position_value = self.position * current_price
        
        # Get timeframe information
        timeframe = getattr(self.live_fetcher.data_loader, 'timeframe', '1h')
        
        return {
            'symbol': self.symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'market_open': self.live_fetcher.is_market_open(),
            'live_mode': self.live_mode,
            'portfolio_value': self.portfolio_value if hasattr(self, 'portfolio_value') else 0,
            'position': self.position if hasattr(self, 'position') else 0,
            'position_value': position_value,
            'balance': self.balance if hasattr(self, 'balance') else 0,
            'current_step': self.current_step if hasattr(self, 'current_step') else 0,
            'total_steps': len(self.data) if hasattr(self, 'data') else 0,
            'last_update': pd.Timestamp.now(),
            'next_update_in': max(0, int((self.last_update_time + self.update_interval) - time.time()))
        }
    
    def __del__(self):
        """Cleanup when environment is destroyed."""
        if hasattr(self, 'live_fetcher'):
            self.stop_live_mode()
