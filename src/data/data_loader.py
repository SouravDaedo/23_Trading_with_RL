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
        self.indicators = self.config['indicators']
        
        # Timeframe configuration
        self.timeframe = self.config['data'].get('timeframe', '1h')
        self.lookback_period = self.config['data'].get('lookback_period', '2y')
        
        # Adjust lookback window based on timeframe
        self.lookback_window = self._calculate_lookback_window(
            self.timeframe,
            self.config['data'].get('lookback_window', 24)
        )
        
    def _calculate_lookback_window(self, timeframe: str, base_window: int) -> int:
        """Calculate appropriate lookback window based on timeframe."""
        # Convert base_window (in hours) to appropriate number of candles for the timeframe
        timeframe_to_hours = {
            '1m': 1/60, '5m': 5/60, '15m': 0.25, '30m': 0.5,
            '1h': 1, '1d': 24, '1wk': 168, '1mo': 720  # Approximate
        }
        
        if timeframe in timeframe_to_hours:
            # Calculate how many candles make up the base window (24 hours)
            return max(1, int((24 / timeframe_to_hours[timeframe]) * (base_window / 24)))
        return base_window  # Fallback to the provided window

    def download_data(self, symbol: str) -> pd.DataFrame:
        """Download historical data for a given symbol with configurable timeframe."""
        try:
            ticker = yf.Ticker(symbol)
            
            # For 1-minute data, we need to use a shorter lookback period
            if self.timeframe == '1m':
                # For 1m data, we can only get up to 7 days of intraday data
                period = '7d'
                interval = '1m'
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(days=7)
            else:
                period = self.lookback_period
                interval = self.timeframe
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(self.lookback_period)
            
            # Download data with the specified timeframe
            data = ticker.history(
                period=period,
                interval=interval,
                prepost=True  # Include pre/post market data for intraday timeframes
            )
            
            if data.empty:
                print(f"No data returned for {symbol} with timeframe {self.timeframe}")
                return pd.DataFrame()
                
            data.index = pd.to_datetime(data.index)
            print(f"Downloaded {len(data)} timesteps for {symbol} at {self.timeframe} intervals")
            return data
        except Exception as e:
            print(f"Error downloading data for {symbol} (timeframe: {self.timeframe}): {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe with optimized parameters for 1m data."""
        # Ensure we have a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Add basic price features
        df['returns'] = df['Close'].pct_change()
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['price_change'] = df['Close'].pct_change()
        
        # Add volume features with shorter windows for 1m data
        df['volume_pct'] = df['Volume'].pct_change()
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()  # 5-min volume MA
        df['volume_ma_15'] = df['Volume'].rolling(window=15).mean()  # 15-min volume MA
        
        # Shorter moving averages for 1m data
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)    # 5-min SMA
        df['SMA_15'] = ta.trend.sma_indicator(df['Close'], window=15)  # 15-min SMA
        df['SMA_60'] = ta.trend.sma_indicator(df['Close'], window=60)  # 1-hour SMA
        df['SMA_240'] = ta.trend.sma_indicator(df['Close'], window=240)  # 4-hour SMA
        
        # EMA for faster response
        df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA_30'] = ta.trend.ema_indicator(df['Close'], window=30)
        
        # RSI with shorter periods
        df['RSI_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD with faster parameters
        fast = 12
        slow = 26
        signal = 9
        macd = ta.trend.MACD(df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Bollinger Bands with tighter parameters
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=1.5)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        # ATR for volatility with shorter window
        df['ATR_7'] = ta.volatility.AverageTrueRange(
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'],
            window=7
        ).average_true_range()
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Order book imbalance (simplified)
        df['price_volume'] = df['Close'] * df['Volume']
        df['order_book_imbalance'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
        
        # Clean up any NaN values
        df = df.dropna()
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data for better training."""
        data = df.copy()
        
        # Normalize price-based indicators
        price_columns = ['Open', 'High', 'Low', 'Close', 'SMA_24', 'SMA_168', 'BB_upper', 'BB_lower', 'BB_middle']
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
        for indicator in ['SMA_24_norm', 'SMA_168_norm', 'RSI_14_norm', 'MACD', 'MACD_signal', 
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
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load data for a given symbol, handling both single file and train/test splits."""
        # Try to load from a single file first
        single_file = f'data/{symbol}.csv'
        if os.path.exists(single_file):
            return pd.read_csv(single_file, index_col=0, parse_dates=True)
        
        # Try to load from train/test split
        train_file = f'data/{symbol}_train.csv'
        test_file = f'data/{symbol}_test.csv'
        if os.path.exists(train_file) and os.path.exists(test_file):
            train = pd.read_csv(train_file, index_col=0, parse_dates=True)
            test = pd.read_csv(test_file, index_col=0, parse_dates=True)
            return pd.concat([train, test]).sort_index()
        
        # If no local data found, try to download it
        print(f"No local data found for {symbol}, attempting to download...")
        data = self.download_data(symbol)
        if data.empty:
            raise FileNotFoundError(f"Could not load or download data for {symbol}")
        
        # Save the downloaded data for future use
        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{symbol}.csv')
        return data
