# DVC Configuration Tracking Guide

This guide shows how to use DVC (Data Version Control) to track model configurations, experiments, and results in your trading system.

## 🚀 Quick Start

### 1. Initialize DVC (if not already done)
```bash
# Install DVC
pip install dvc[all]

# Initialize DVC in your project
dvc init
```

### 2. Run Experiments

#### Single Experiment
```bash
python run_dvc_experiments.py --action single
```

#### Hyperparameter Sweep
```bash
python run_dvc_experiments.py --action sweep
```

#### Architecture Comparison
```bash
python run_dvc_experiments.py --action architecture
```

#### Agent Type Comparison
```bash
python run_dvc_experiments.py --action agent_comparison
```

### 3. Analyze Results
```bash
python run_dvc_experiments.py --action analyze
```

## 📊 DVC Pipeline Structure

### Pipeline Stages (`dvc.yaml`)
- **data_preparation**: Load and preprocess market data
- **train_dqn**: Train DQN agents for each symbol
- **train_sac**: Train SAC agents for each symbol  
- **train_multi_agent**: Train multi-agent system
- **evaluate_models**: Evaluate trained models

### Parameters (`params.yaml`)
All model configurations are tracked in `params.yaml`:
- Data parameters (symbols, dates, intervals)
- Environment settings (balance, costs, rewards)
- Model architecture (layers, activation, dropout)
- Training parameters (episodes, learning rate, batch size)
- Multi-agent settings (agent types, allocation rules)

## 🔬 Experiment Tracking

### Creating Experiments
```python
from experiments.experiment_tracker import DVCExperimentTracker

tracker = DVCExperimentTracker()

# Create experiment with custom config
exp_id = tracker.create_experiment(
    name="high_lr_experiment",
    description="Testing higher learning rates",
    tags=["hyperparameter", "learning_rate"],
    config_overrides={
        'training': {'learning_rate': 0.001},
        'sac': {'learning_rate': 0.001}
    }
)

# Run experiment
results = tracker.run_experiment(exp_id)
```

### Comparing Experiments
```python
# Compare multiple experiments
comparison_df = tracker.compare_experiments(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    metrics=['portfolio_total_return', 'sharpe_ratio']
)

# Find best experiment
best_exp = tracker.get_best_experiment('portfolio_total_return')
```

## 📈 Metrics Tracking

### Automatic Metrics
The system automatically tracks:
- **Portfolio metrics**: Total return, Sharpe ratio, max drawdown
- **Agent metrics**: Win rate, average reward, trade count
- **Training metrics**: Episode rewards, convergence, losses
- **Allocation metrics**: Entropy, diversity, stability

### Custom Metrics
```python
from src.utils.metrics_logger import DVCMetricsLogger

logger = DVCMetricsLogger("my_experiment")

# Log custom metrics
logger.log_custom_metric("custom_score", 0.85)
logger.log_training_step(episode=100, metrics={
    'reward': 0.05,
    'portfolio_value': 11000
})

# Save for DVC
logger.save_for_dvc()
```

## 🗂️ File Structure

```
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # All parameters (tracked by DVC)
├── experiments/
│   ├── experiment_tracker.py   # Experiment management
│   └── {experiment_id}/        # Individual experiment configs
├── metrics/
│   ├── metrics.json            # Main DVC metrics file
│   └── {experiment}_metrics.json # Detailed experiment metrics
├── models/
│   ├── dqn_{symbol}/          # DQN model outputs
│   ├── sac_{symbol}/          # SAC model outputs
│   └── multi_agent/           # Multi-agent system models
└── plots/
    └── {experiment}/          # Training plots and visualizations
```

## 🔄 DVC Commands

### Run Pipeline
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train_multi_agent

# Run with different parameters
dvc repro -f  # Force re-run
```

### Track Experiments
```bash
# Show experiment metrics
dvc metrics show

# Compare experiments
dvc metrics diff

# Show plots
dvc plots show

# List all experiments
dvc exp show
```

### Parameter Management
```bash
# Show current parameters
dvc params diff

# Run with parameter override
dvc exp run -S training.learning_rate=0.001
```

## 🎯 Common Use Cases

### 1. Hyperparameter Tuning
```python
# Create parameter grid
param_grid = {
    'learning_rates': [0.0001, 0.0003, 0.001],
    'batch_sizes': [32, 64, 128]
}

# Run sweep
experiment_ids = create_hyperparameter_sweep()
```

### 2. Model Architecture Search
```python
architectures = [
    {'hidden_layers': [128, 64], 'activation': 'relu'},
    {'hidden_layers': [256, 128, 64], 'activation': 'tanh'},
    {'hidden_layers': [512, 256, 128], 'activation': 'leaky_relu'}
]

experiment_ids = run_model_comparison()
```

### 3. Agent Type Comparison
```python
# Compare DQN vs SAC vs Mixed
config_overrides = {
    'multi_agent': {
        'agent_types': ['dqn', 'dqn', 'dqn', 'dqn']  # All DQN
    }
}
```

## 📊 Metrics Explanation

### Portfolio Metrics
- **total_return**: Overall portfolio return percentage
- **sharpe_ratio**: Risk-adjusted return measure
- **max_drawdown**: Maximum portfolio decline from peak
- **volatility**: Portfolio return standard deviation
- **calmar_ratio**: Return vs max drawdown ratio

### Agent Metrics  
- **win_rate**: Percentage of profitable trades
- **avg_reward**: Average reward per episode
- **total_trades**: Number of trades executed
- **agent_return**: Individual agent performance

### Training Metrics
- **episode_rewards**: Reward progression during training
- **portfolio_values**: Portfolio value over time
- **allocation_entropy**: Diversification measure
- **convergence_rate**: Learning speed indicator

## 🔧 Configuration Tips

### 1. Parameter Organization
Keep related parameters grouped in `params.yaml`:
```yaml
training:
  episodes: 1000
  learning_rate: 0.0001
  batch_size: 32

model:
  hidden_layers: [256, 128, 64]
  activation: "relu"
```

### 2. Experiment Naming
Use descriptive names and tags:
```python
exp_id = tracker.create_experiment(
    name="sac_high_lr_long_training",
    description="SAC agents with 0.001 LR for 2000 episodes",
    tags=["sac", "high_lr", "long_training"]
)
```

### 3. Metrics Selection
Focus on key business metrics:
```python
key_metrics = [
    'portfolio_total_return',
    'portfolio_sharpe_ratio', 
    'portfolio_max_drawdown',
    'allocation_entropy'
]
```

## 🚨 Best Practices

1. **Version Control**: Commit `dvc.yaml` and `params.yaml` to git
2. **Reproducibility**: Always specify random seeds in config
3. **Documentation**: Add clear descriptions to experiments
4. **Cleanup**: Archive old/failed experiments regularly
5. **Monitoring**: Check metrics during long training runs
6. **Comparison**: Always compare against baseline experiments

## 🛠️ Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `pip install dvc[all]`
2. **Pipeline errors**: Check file paths in `dvc.yaml`
3. **Metrics not saving**: Ensure metrics directory exists
4. **Experiment conflicts**: Use unique experiment names

### Debug Commands
```bash
# Check DVC status
dvc status

# Validate pipeline
dvc dag

# Show detailed logs
dvc repro --verbose
```

This setup provides comprehensive experiment tracking, reproducibility, and comparison capabilities for your trading system development.
