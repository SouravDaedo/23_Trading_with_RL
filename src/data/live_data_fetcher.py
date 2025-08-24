import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading
import queue
from .data_loader import DataLoader

class LiveDataFetcher:
    """Fetches real-time market data for live trading simulation."""
    
    def __init__(self, symbol: str, config_path: str = "config/config.yaml"):
        self.symbol = symbol
        self.data_loader = DataLoader(config_path)
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Initialize with recent historical data
        self.historical_data = self._get_recent_historical_data()
        self.current_data = self.historical_data.copy()
        
    def _get_recent_historical_data(self) -> pd.DataFrame:
        """Get recent historical data to initialize the environment."""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get data using the configured timeframe
            data = ticker.history(
                period=self.data_loader.lookback_period,
                interval=self.data_loader.timeframe,
                prepost=True
            )
            
            if data.empty:
                print(f"Warning: No historical data found for {self.symbol} with timeframe {self.data_loader.timeframe}")
                return pd.DataFrame()
                
            return data.dropna()
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
        
        # Add technical indicators
        data = self.data_loader.add_technical_indicators(data)
        return data.dropna()
    
    def _fetch_latest_data(self) -> Optional[pd.Series]:
        """Fetch the latest market data point."""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get data using the configured timeframe
            data = ticker.history(
                period='1d',  # Get last day of data to ensure we have the latest candle
                interval=self.data_loader.timeframe
            )
            
            if not data.empty:
                # Get the most recent data point
                latest = data.iloc[-1]
                latest.name = data.index[-1]
                return latest
        except Exception as e:
            print(f"Error fetching latest data: {e}")
        return None
    
    def start_streaming(self, update_interval: int = None):
        """Start streaming real-time data.
        
        Args:
            update_interval: Time in seconds between updates. If None, will be set based on timeframe.
        """
        if update_interval is None:
            # Set default update interval based on timeframe
            timeframe_seconds = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '1d': 86400, '1wk': 604800, '1mo': 2592000
            }
            update_interval = timeframe_seconds.get(self.data_loader.timeframe, 3600)
            
        self.running = True
        self.thread = threading.Thread(
            target=self._stream_worker, 
            args=(update_interval,),
            daemon=True
        )
        self.thread.start()
        print(f"Started live data streaming for {self.symbol} at {self.data_loader.timeframe} timeframe")
    
    def _stream_worker(self, update_interval: int):
        """Worker thread for continuous data fetching."""
        while self.running:
            latest_data = self._fetch_latest_data()
            if latest_data is not None:
                # Add technical indicators to the latest data point
                temp_df = pd.concat([self.current_data.tail(50), latest_data.to_frame().T])
                temp_df = self.data_loader.add_technical_indicators(temp_df)
                
                if not temp_df.empty:
                    latest_with_indicators = temp_df.iloc[-1]
                    self.data_queue.put(latest_with_indicators)
                    
                    # Update current data
                    self.current_data = pd.concat([
                        self.current_data, 
                        latest_with_indicators.to_frame().T
                    ]).tail(1000)  # Keep last 1000 data points
            
            time.sleep(update_interval)
    
    def get_latest_data(self) -> Optional[pd.Series]:
        """Get the latest data point from the queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_dataset(self) -> pd.DataFrame:
        """Get the current complete dataset."""
        return self.current_data.copy()
    
    def stop_streaming(self):
        """Stop the data streaming."""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Stopped live data streaming")
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open (basic US market hours).
        
        Note: For daily or weekly timeframes, we consider the market open on weekdays.
        """
        now = datetime.now()
        
        # For daily or weekly timeframes, check if it's a weekday
        if self.data_loader.timeframe in ['1d', '1wk', '1mo']:
            return now.weekday() < 5  # Monday=0, Sunday=6
            
        # For intraday timeframes, check market hours
        if now.weekday() >= 5:  # Weekend
            return False
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
