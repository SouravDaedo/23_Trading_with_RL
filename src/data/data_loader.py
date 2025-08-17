import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import List, Tuple, Dict
import yaml
import os

class DataLoader:
    """Handles loading and preprocessing of historical market data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.symbols = self.config['data']['symbols']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.lookback_window = self.config['data']['lookback_window']
        self.indicators = self.config['indicators']
        
        # Timestep configuration
        self.interval = self.config['data'].get('interval', '1d')
        self.period = self.config['data'].get('period', 'max')
        
    def download_data(self, symbol: str) -> pd.DataFrame:
        """Download historical data for a given symbol with configurable timestep."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Use interval for timestep granularity
            if self.start_date and self.end_date:
                data = ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)
            else:
                data = ticker.history(period=self.period, interval=self.interval)
            
            data.index = pd.to_datetime(data.index)
            print(f"Downloaded {len(data)} timesteps for {symbol} at {self.interval} intervals")
            return data
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        data = df.copy()
        
        # Simple Moving Averages
        if "SMA_10" in self.indicators:
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        if "SMA_20" in self.indicators:
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        
        # RSI
        if "RSI_14" in self.indicators:
            data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)
        
        # MACD
        if "MACD" in self.indicators:
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        if "BB_upper" in self.indicators or "BB_lower" in self.indicators:
            bb = ta.volatility.BollingerBands(data['Close'])
            if "BB_upper" in self.indicators:
                data['BB_upper'] = bb.bollinger_hband()
            if "BB_lower" in self.indicators:
                data['BB_lower'] = bb.bollinger_lband()
            data['BB_middle'] = bb.bollinger_mavg()
        
        # Volume indicators
        if "volume_sma" in self.indicators:
            data['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume'])
        
        # Price change and returns
        data['price_change'] = data['Close'].pct_change()
        data['high_low_pct'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data for better training."""
        data = df.copy()
        
        # Normalize price-based indicators
        price_columns = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20', 'BB_upper', 'BB_lower', 'BB_middle']
        for col in price_columns:
            if col in data.columns:
                data[f'{col}_norm'] = data[col] / data['Close'].rolling(window=20).mean()
        
        # RSI is already normalized (0-100)
        if 'RSI_14' in data.columns:
            data['RSI_14_norm'] = data['RSI_14'] / 100.0
        
        # Normalize volume
        if 'Volume' in data.columns:
            data['Volume_norm'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        return data
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for the RL agent."""
        data = df.copy()
        
        # Select relevant features
        feature_columns = []
        
        # Price features
        if 'Close_norm' in data.columns:
            feature_columns.append('Close_norm')
        if 'high_low_pct' in data.columns:
            feature_columns.append('high_low_pct')
        if 'price_change' in data.columns:
            feature_columns.append('price_change')
        
        # Technical indicators
        for indicator in ['SMA_10_norm', 'SMA_20_norm', 'RSI_14_norm', 'MACD', 'MACD_signal', 
                         'BB_upper_norm', 'BB_lower_norm', 'Volume_norm']:
            if indicator in data.columns:
                feature_columns.append(indicator)
        
        # Create lagged features
        for lag in range(1, min(5, self.lookback_window)):
            for col in ['price_change', 'Volume_norm']:
                if col in data.columns:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                    feature_columns.append(f'{col}_lag_{lag}')
        
        # Select only feature columns
        features = data[feature_columns].copy()
        
        # Forward fill and backward fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def prepare_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data preparation pipeline."""
        print(f"Preparing data for {symbol}...")
        
        # Download raw data
        raw_data = self.download_data(symbol)
        if raw_data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Add technical indicators
        data_with_indicators = self.add_technical_indicators(raw_data)
        
        # Normalize data
        normalized_data = self.normalize_data(data_with_indicators)
        
        # Create features
        features = self.create_features(normalized_data)
        
        # Split into train and test
        split_idx = int(len(features) * self.config['data']['train_split'])
        
        train_data = features.iloc[:split_idx].copy()
        test_data = features.iloc[split_idx:].copy()
        
        print(f"Data prepared: {len(train_data)} training samples, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to file."""
        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{filename}', index=True)
        print(f"Data saved to data/{filename}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from file."""
        return pd.read_csv(f'data/{filename}', index_col=0, parse_dates=True)
