# 📈 Reinforcement Learning Trading System

## 🎯 Project Overview

This repository contains a comprehensive **Reinforcement Learning (RL) trading system** that uses **Deep Q-Network (DQN)** algorithms to learn optimal trading strategies from historical market data. The system supports both **historical backtesting** and **real-time live trading** with paper trading capabilities.

### 🚀 Key Features

- **🤖 Deep Q-Network (DQN) Agent**: Advanced RL agent for trading decisions
- **📊 Custom Trading Environment**: Gymnasium-compatible environment with realistic trading mechanics
- **📈 Technical Analysis**: 7+ technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- **🔴 Live Trading**: Real-time data streaming and paper trading capabilities
- **📉 Comprehensive Backtesting**: Performance evaluation with multiple metrics
- **⚙️ Configurable**: YAML-based configuration for all parameters
- **📱 Visualization**: Trading performance plots and learning curves
- **🎯 Multi-Asset Support**: Trade multiple stocks simultaneously

---

## 🏗️ Project Architecture

```
23_Trading_with_RL/
├── 📁 src/                          # Core source code
│   ├── 📁 agents/                   # RL agents implementation
│   │   └── dqn_agent.py            # Deep Q-Network agent
│   ├── 📁 data/                     # Data handling modules
│   │   ├── data_loader.py          # Historical data fetching
│   │   └── live_data_fetcher.py    # Real-time data streaming
│   ├── 📁 environment/              # Trading environments
│   │   ├── trading_env.py          # Base trading environment
│   │   └── live_trading_env.py     # Live trading environment
│   ├── 📁 utils/                    # Utility functions
│   │   └── metrics.py              # Performance metrics
│   └── 📁 visualization/            # Plotting and analysis
│       └── plotting.py             # Trading visualizations
├── 📁 config/                       # Configuration files
│   └── config.yaml                 # Main configuration
├── 📁 notebooks/                    # Jupyter notebooks
├── 📁 models/                       # Saved model checkpoints
├── 📁 logs/                         # Training logs
├── 📁 data/                         # Historical data storage
├── train_agent.py                  # Main training script
├── evaluate_agent.py               # Model evaluation script
├── live_test.py                    # Live trading test script
├── requirements.txt                # Python dependencies
└── README.md                       # Basic project info
```

---

## 🧠 Core Components

### 1. **DQN Agent** (`src/agents/dqn_agent.py`)
- **Architecture**: Multi-layer neural network (256→128→64 neurons)
- **Algorithm**: Deep Q-Network with experience replay
- **Features**: 
  - Epsilon-greedy exploration strategy
  - Target network for stable training
  - Experience replay buffer (10,000 transitions)
  - Configurable learning parameters

### 2. **Trading Environment** (`src/environment/trading_env.py`)
- **Action Space**: 3 discrete actions (Hold, Buy, Sell)
- **Observation Space**: Technical indicators + portfolio state
- **Reward Function**: Portfolio value changes with transaction costs
- **Features**:
  - Realistic transaction costs (0.1% per trade)
  - Position limits and risk management
  - Comprehensive state representation

### 3. **Live Trading Environment** (`src/environment/live_trading_env.py`)
- **Real-time Data**: Continuous market data updates
- **Market Hours**: Automatic detection of trading hours
- **Paper Trading**: Risk-free live strategy testing
- **Features**:
  - Configurable update intervals (1 minute to hours)
  - Automatic data refresh during trading
  - Live portfolio tracking

### 4. **Data Management**
- **Historical Data** (`src/data/data_loader.py`):
  - Yahoo Finance integration
  - Technical indicator calculation
  - Data preprocessing and normalization
  
- **Live Data** (`src/data/live_data_fetcher.py`):
  - Real-time price streaming
  - Technical indicator updates
  - Market status monitoring

---

## 📊 Technical Indicators

The system implements 7+ technical indicators for market analysis:

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **SMA_10** | Simple Moving Average | 10-day window |
| **SMA_20** | Simple Moving Average | 20-day window |
| **RSI_14** | Relative Strength Index | 14-day period |
| **MACD** | Moving Average Convergence Divergence | 12, 26, 9 |
| **BB_upper** | Bollinger Bands Upper | 20-day, 2 std |
| **BB_lower** | Bollinger Bands Lower | 20-day, 2 std |
| **Volume_SMA** | Volume Moving Average | 20-day window |

---

## 🎮 Usage Guide

### 🏋️ Training a New Agent

```bash
# Train with default configuration
python train_agent.py

# The training process will:
# 1. Load historical data for configured symbols
# 2. Create trading environment with technical indicators
# 3. Initialize DQN agent with neural network
# 4. Train for 1000 episodes with experience replay
# 5. Save best model to models/ directory
# 6. Generate training plots and metrics
```

### 📈 Evaluating Trained Models

```bash
# Evaluate the latest trained model
python evaluate_agent.py

# Features:
# - Backtesting on unseen test data
# - Performance metrics calculation
# - Comparison with buy-and-hold benchmark
# - Detailed trading analysis plots
```

### 🔴 Live Trading Testing

```bash
# Quick live test (50 steps, 1-minute updates)
python live_test.py --symbol AAPL --steps 50 --interval 60

# Paper trading mode (2 hours continuous)
python live_test.py --mode paper --symbol TSLA --duration 2

# Advanced options
python live_test.py --symbol GOOGL --model models/best_model.pth --steps 100 --interval 300
```

---

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Asset Selection
data:
  symbols: ["AAPL", "GOOGL", "MSFT", "TSLA"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  train_split: 0.8
  lookback_window: 20

# Trading Parameters
environment:
  initial_balance: 10000
  transaction_cost: 0.001  # 0.1% per trade
  max_position: 1.0        # 100% max position
  reward_scaling: 1.0

# Neural Network Architecture
model:
  hidden_layers: [256, 128, 64]
  activation: "relu"
  dropout: 0.2

# Training Hyperparameters
training:
  episodes: 1000
  batch_size: 32
  learning_rate: 0.0001
  gamma: 0.95              # Discount factor
  epsilon_start: 1.0       # Exploration rate
  epsilon_end: 0.01
  epsilon_decay: 0.995
  memory_size: 10000       # Replay buffer size
  target_update: 100       # Target network update frequency
```

---

## 📊 Performance Metrics

The system tracks comprehensive trading performance metrics:

### 📈 Return Metrics
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Year-over-year performance
- **Excess Return**: Performance vs benchmark

### 📉 Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Potential losses at confidence levels

### 🎯 Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Mean profit per trade
- **Trade Frequency**: Number of trades per period

---

## 🔬 Research & Experimentation

### 📓 Jupyter Notebooks
- **Data Analysis**: Exploratory data analysis of market data
- **Strategy Development**: Prototype and test new trading strategies
- **Hyperparameter Tuning**: Optimize model parameters
- **Performance Analysis**: Deep dive into trading results

### 🧪 Experimental Features
- **Multi-Asset Trading**: Simultaneous trading across multiple stocks
- **Advanced Indicators**: Custom technical analysis tools
- **Risk Management**: Position sizing and stop-loss mechanisms
- **Market Regime Detection**: Adapt strategies to market conditions

---

## 🚀 Getting Started

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd 23_Trading_with_RL

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, gymnasium, yfinance; print('✅ All dependencies installed')"
```

### 2. **Quick Start Training**
```bash
# Train your first agent
python train_agent.py

# This will:
# - Download historical data for AAPL, GOOGL, MSFT, TSLA
# - Train DQN agent for 1000 episodes
# - Save best model to models/
# - Generate performance plots
```

### 3. **Evaluate Performance**
```bash
# Test the trained agent
python evaluate_agent.py

# View results in:
# - Console output (metrics summary)
# - Generated plots (trading performance)
# - Logs directory (detailed logs)
```

### 4. **Live Testing**
```bash
# Test with real-time data
python live_test.py --symbol AAPL --steps 20 --interval 60

# Monitor live performance:
# - Real-time price updates
# - Trading decisions
# - Portfolio value changes
# - Performance metrics
```

---

## 🛠️ Development & Customization

### 🎯 Adding New Trading Strategies
1. Extend the `DQNAgent` class in `src/agents/`
2. Implement custom reward functions in `TradingEnvironment`
3. Add new technical indicators in `DataLoader`

### 📊 Custom Technical Indicators
```python
# Add to src/data/data_loader.py
def add_custom_indicator(self, df):
    # Your custom indicator logic
    df['custom_indicator'] = your_calculation(df)
    return df
```

### 🔧 Environment Modifications
- **Action Space**: Modify buy/sell/hold actions
- **Observation Space**: Add new market features
- **Reward Function**: Implement custom reward logic
- **Risk Management**: Add position limits and stops

---

## 📚 Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **Gymnasium**: RL environment interface
- **yfinance**: Market data provider
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

### Technical Analysis
- **TA-Lib**: Technical analysis library
- **pandas-ta**: Additional indicators

### Development Tools
- **Jupyter**: Interactive development
- **TensorBoard**: Training monitoring
- **PyYAML**: Configuration management

---

## 🤝 Contributing

### 🐛 Bug Reports
- Use GitHub issues for bug reports
- Include error messages and system info
- Provide minimal reproduction steps

### 💡 Feature Requests
- Suggest new trading strategies
- Propose technical indicators
- Request performance improvements

### 🔧 Development Guidelines
- Follow PEP 8 coding standards
- Add unit tests for new features
- Update documentation for changes
- Use meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenAI Gym/Gymnasium**: RL environment framework
- **Yahoo Finance**: Market data provider
- **PyTorch Community**: Deep learning tools
- **QuantLib**: Financial mathematics library

---

## 📞 Support

For questions, issues, or contributions:
- 📧 Create GitHub issues for bugs
- 💬 Use discussions for questions
- 📖 Check documentation first
- 🔍 Search existing issues

---

**Happy Trading! 📈🤖**
