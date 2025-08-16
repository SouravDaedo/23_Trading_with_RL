# RL Trading Agent

A reinforcement learning trading agent that learns from historical market data using Deep Q-Network (DQN).

## Features

- **Historical Data Loading**: Fetches and preprocesses stock market data
- **Technical Indicators**: Implements various technical analysis indicators
- **RL Environment**: Custom Gym environment for trading simulation
- **DQN Agent**: Deep Q-Network implementation for trading decisions
- **Backtesting**: Comprehensive performance evaluation
- **Visualization**: Trading performance and learning curves

## Project Structure

```
├── data/                   # Historical data storage
├── src/
│   ├── environment/        # Trading environment
│   ├── agents/            # RL agents
│   ├── data/              # Data loading and preprocessing
│   ├── utils/             # Utility functions
│   └── visualization/     # Plotting and analysis
├── notebooks/             # Jupyter notebooks for experimentation
├── models/               # Saved model checkpoints
├── logs/                 # Training logs and tensorboard
└── config/               # Configuration files
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python train_agent.py
```

3. Evaluate the trained agent:
```bash
python evaluate_agent.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Stock symbols to trade
- Training parameters
- Model architecture
- Risk management settings

## Usage Examples

See `notebooks/` for detailed examples and tutorials.
