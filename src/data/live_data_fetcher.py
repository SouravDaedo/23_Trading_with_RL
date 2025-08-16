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
        
    def _get_recent_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get recent historical data to initialize the environment."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1m')
        
        if data.empty:
            # Fallback to daily data if minute data is not available
            data = ticker.history(start=start_date, end=end_date, interval='1d')
        
        # Add technical indicators
        data = self.data_loader.add_technical_indicators(data)
        return data.dropna()
    
    def _fetch_latest_data(self) -> Optional[pd.Series]:
        """Fetch the latest market data point."""
        try:
            ticker = yf.Ticker(self.symbol)
            # Get the last few minutes of data
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                latest = data.iloc[-1]
                latest.name = data.index[-1]
                return latest
        except Exception as e:
            print(f"Error fetching latest data: {e}")
        return None
    
    def start_streaming(self, update_interval: int = 60):
        """Start streaming real-time data."""
        self.running = True
        self.thread = threading.Thread(
            target=self._stream_worker, 
            args=(update_interval,)
        )
        self.thread.start()
        print(f"Started live data streaming for {self.symbol}")
    
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
        """Check if the market is currently open (basic US market hours)."""
        now = datetime.now()
        # Basic check for US market hours (9:30 AM - 4:00 PM EST, Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
