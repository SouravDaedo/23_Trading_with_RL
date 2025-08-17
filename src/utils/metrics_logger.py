"""
Metrics Logger for DVC Integration
Logs training and evaluation metrics in DVC-compatible format.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class MetricsLogger:
    """Logs metrics for DVC tracking and experiment comparison."""
    
    def __init__(self, experiment_id: str = None, metrics_dir: str = "metrics"):
        """
        Initialize metrics logger.
        
        Args:
            experiment_id: Unique experiment identifier
            metrics_dir: Directory to save metrics files
        """
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        self.metrics = {}
        self.training_history = []
        self.evaluation_results = {}
        
    def log_training_step(self, episode: int, metrics: Dict[str, float]):
        """Log metrics for a training step."""
        step_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_history.append(step_data)
        
    def log_training_summary(self, metrics: Dict[str, float]):
        """Log overall training summary metrics."""
        self.metrics.update({
            f"training_{key}": value for key, value in metrics.items()
        })
        
    def log_evaluation_results(self, agent_type: str, symbol: str, 
                             results: Dict[str, float]):
        """Log evaluation results for a specific agent/symbol."""
        key = f"{agent_type}_{symbol}"
        self.evaluation_results[key] = results
        
        # Add to main metrics with prefixes
        for metric, value in results.items():
            self.metrics[f"eval_{key}_{metric}"] = value
    
    def log_multi_agent_results(self, results: Dict[str, Any]):
        """Log multi-agent system results."""
        # Portfolio-level metrics
        if 'portfolio_metrics' in results:
            portfolio = results['portfolio_metrics']
            for key, value in portfolio.items():
                self.metrics[f"portfolio_{key}"] = value
        
        # Individual agent metrics
        if 'stock_agents' in results:
            for symbol, agent_stats in results['stock_agents'].items():
                for metric, value in agent_stats.items():
                    self.metrics[f"agent_{symbol}_{metric}"] = value
        
        # Allocation metrics
        if 'allocation_history' in results and results['allocation_history']:
            allocations = results['allocation_history'][-10:]  # Last 10 allocations
            avg_allocation = np.mean([a['allocation'] for a in allocations], axis=0)
            allocation_entropy = -np.sum(avg_allocation * np.log(avg_allocation + 1e-8))
            
            self.metrics['allocation_entropy'] = allocation_entropy
            self.metrics['allocation_std'] = np.std(avg_allocation)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters used in the experiment."""
        # Flatten nested parameters
        flat_params = self._flatten_dict(params)
        
        # Add to metrics with hp_ prefix
        for key, value in flat_params.items():
            if isinstance(value, (int, float, str, bool)):
                self.metrics[f"hp_{key}"] = value
    
    def log_custom_metric(self, name: str, value: float, step: int = None):
        """Log a custom metric."""
        if step is not None:
            # Time series metric
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({'step': step, 'value': value})
        else:
            # Scalar metric
            self.metrics[name] = value
    
    def calculate_trading_metrics(self, portfolio_values: List[float], 
                                returns: List[float]) -> Dict[str, float]:
        """Calculate comprehensive trading metrics."""
        if not portfolio_values or not returns:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns)
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = avg_return / (volatility + 1e-8)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = avg_return / (downside_std + 1e-8)
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(returns)
        }
    
    def save_metrics(self, filename: str = None):
        """Save metrics to JSON file for DVC."""
        if filename is None:
            filename = f"{self.experiment_id}_metrics.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Add metadata
        output_data = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'training_history_length': len(self.training_history),
            'evaluation_results': self.evaluation_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"📊 Metrics saved to: {filepath}")
        return filepath
    
    def save_training_history(self, filename: str = None):
        """Save detailed training history to CSV."""
        if not self.training_history:
            return
        
        if filename is None:
            filename = f"{self.experiment_id}_training_history.csv"
        
        filepath = os.path.join(self.metrics_dir, filename)
        df = pd.DataFrame(self.training_history)
        df.to_csv(filepath, index=False)
        
        print(f"📈 Training history saved to: {filepath}")
        return filepath
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the experiment."""
        if not self.training_history:
            return self.metrics
        
        df = pd.DataFrame(self.training_history)
        
        summary = {}
        
        # Training progression metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'episode':
                summary[f"{col}_final"] = df[col].iloc[-1]
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
                summary[f"{col}_max"] = df[col].max()
                summary[f"{col}_min"] = df[col].min()
        
        # Combine with existing metrics
        summary.update(self.metrics)
        
        return summary
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', 
                     sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                # Handle numeric lists by taking mean
                items.append((new_key, np.mean(v)))
            elif isinstance(v, (int, float, str, bool)):
                items.append((new_key, v))
        return dict(items)

class DVCMetricsLogger(MetricsLogger):
    """Extended metrics logger with DVC-specific functionality."""
    
    def __init__(self, experiment_id: str = None, metrics_dir: str = "metrics"):
        super().__init__(experiment_id, metrics_dir)
        self.dvc_metrics_file = os.path.join(metrics_dir, "metrics.json")
    
    def save_for_dvc(self):
        """Save metrics in DVC-compatible format."""
        # Save main metrics file for DVC
        with open(self.dvc_metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save detailed experiment metrics
        self.save_metrics()
        self.save_training_history()
        
        print(f"📊 DVC metrics saved to: {self.dvc_metrics_file}")
    
    def log_dvc_plot_data(self, plot_name: str, data: Dict[str, List]):
        """Log data for DVC plots."""
        plots_dir = os.path.join(self.metrics_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_file = os.path.join(plots_dir, f"{plot_name}.json")
        with open(plot_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📈 Plot data saved to: {plot_file}")

# Utility functions for common use cases
def create_metrics_logger_from_config(config_path: str, 
                                     experiment_name: str = None) -> DVCMetricsLogger:
    """Create metrics logger from configuration file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_id = experiment_name or config.get('experiment', {}).get('name', 'default')
    logger = DVCMetricsLogger(experiment_id)
    
    # Log hyperparameters from config
    logger.log_hyperparameters(config)
    
    return logger

def log_agent_performance(logger: MetricsLogger, agent_name: str, 
                         episode_rewards: List[float], 
                         portfolio_values: List[float]):
    """Log agent performance metrics."""
    if not episode_rewards or not portfolio_values:
        return
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate trading metrics
    trading_metrics = logger.calculate_trading_metrics(portfolio_values, returns)
    
    # Add agent-specific prefix
    agent_metrics = {f"{agent_name}_{k}": v for k, v in trading_metrics.items()}
    
    # Add episode-based metrics
    agent_metrics.update({
        f"{agent_name}_avg_episode_reward": np.mean(episode_rewards),
        f"{agent_name}_std_episode_reward": np.std(episode_rewards),
        f"{agent_name}_final_portfolio_value": portfolio_values[-1],
        f"{agent_name}_total_episodes": len(episode_rewards)
    })
    
    # Log to main metrics
    for key, value in agent_metrics.items():
        logger.log_custom_metric(key, value)
