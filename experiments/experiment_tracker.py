"""
DVC Experiment Tracking System
Tracks model configurations, metrics, and results using DVC.
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess
import shutil

class DVCExperimentTracker:
    """Tracks experiments using DVC for reproducibility and versioning."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize DVC experiment tracker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.experiments_dir = os.path.join(project_root, "experiments")
        self.metrics_dir = os.path.join(project_root, "metrics")
        self.plots_dir = os.path.join(project_root, "plots")
        self.models_dir = os.path.join(project_root, "models")
        
        # Create directories
        for dir_path in [self.experiments_dir, self.metrics_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_experiment(self, name: str, description: str = "", 
                         tags: List[str] = None, 
                         config_overrides: Dict[str, Any] = None) -> str:
        """
        Create a new experiment with unique configuration.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: List of tags for the experiment
            config_overrides: Parameter overrides for this experiment
            
        Returns:
            experiment_id: Unique experiment identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Load base parameters
        params_file = os.path.join(self.project_root, "params.yaml")
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        # Apply overrides
        if config_overrides:
            params = self._deep_update(params, config_overrides)
        
        # Update experiment metadata
        params['experiment'] = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        }
        
        # Save experiment parameters
        exp_params_file = os.path.join(exp_dir, "params.yaml")
        with open(exp_params_file, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        
        # Create experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'config_file': exp_params_file,
            'metrics_file': os.path.join(self.metrics_dir, f"{experiment_id}_metrics.json"),
            'plots_dir': os.path.join(self.plots_dir, experiment_id)
        }
        
        metadata_file = os.path.join(exp_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Created experiment: {experiment_id}")
        print(f"   Description: {description}")
        print(f"   Tags: {tags}")
        print(f"   Config: {exp_params_file}")
        
        return experiment_id
    
    def run_experiment(self, experiment_id: str, stage: str = None) -> Dict[str, Any]:
        """
        Run an experiment using DVC pipeline.
        
        Args:
            experiment_id: Experiment identifier
            stage: Specific stage to run (optional)
            
        Returns:
            results: Experiment results
        """
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update experiment status
        self._update_experiment_status(experiment_id, 'running')
        
        # Copy experiment params to root for DVC
        exp_params = os.path.join(exp_dir, "params.yaml")
        root_params = os.path.join(self.project_root, "params.yaml")
        shutil.copy2(exp_params, root_params)
        
        try:
            # Run DVC pipeline
            cmd = ["dvc", "repro"]
            if stage:
                cmd.extend(["-s", stage])
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                self._update_experiment_status(experiment_id, 'completed')
                print(f" Experiment {experiment_id} completed successfully")
            else:
                self._update_experiment_status(experiment_id, 'failed')
                print(f" Experiment {experiment_id} failed:")
                print(result.stderr)
                
            # Collect results
            results = self._collect_experiment_results(experiment_id)
            return results
            
        except Exception as e:
            self._update_experiment_status(experiment_id, 'failed')
            print(f" Experiment {experiment_id} failed: {e}")
            return {}
    
    def compare_experiments(self, experiment_ids: List[str], 
                          metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            comparison_df: DataFrame with experiment comparison
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_dir = os.path.join(self.experiments_dir, exp_id)
            
            # Load metadata
            metadata_file = os.path.join(exp_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                continue
            
            # Load metrics
            metrics_file = metadata.get('metrics_file')
            if metrics_file and os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    exp_metrics = json.load(f)
            else:
                exp_metrics = {}
            
            # Load parameters
            params_file = os.path.join(exp_dir, "params.yaml")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = yaml.safe_load(f)
            else:
                params = {}
            
            # Combine data
            row_data = {
                'experiment_id': exp_id,
                'name': metadata.get('name', ''),
                'status': metadata.get('status', ''),
                'created_at': metadata.get('created_at', ''),
                'tags': ', '.join(metadata.get('tags', []))
            }
            
            # Add key parameters
            if 'training' in params:
                row_data.update({
                    'episodes': params['training'].get('episodes'),
                    'learning_rate': params['training'].get('learning_rate'),
                    'batch_size': params['training'].get('batch_size')
                })
            
            # Add metrics
            if metrics:
                for metric in metrics:
                    row_data[metric] = exp_metrics.get(metric)
            else:
                row_data.update(exp_metrics)
            
            comparison_data.append(row_data)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def get_best_experiment(self, metric: str, 
                           minimize: bool = False) -> Optional[str]:
        """
        Find the best experiment based on a metric.
        
        Args:
            metric: Metric to optimize
            minimize: Whether to minimize the metric (default: maximize)
            
        Returns:
            best_experiment_id: ID of the best experiment
        """
        all_experiments = self._get_all_experiments()
        best_value = float('inf') if minimize else float('-inf')
        best_experiment = None
        
        for exp_id in all_experiments:
            metrics_file = os.path.join(
                self.metrics_dir, f"{exp_id}_metrics.json"
            )
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                if metric in metrics:
                    value = metrics[metric]
                    if minimize and value < best_value:
                        best_value = value
                        best_experiment = exp_id
                    elif not minimize and value > best_value:
                        best_value = value
                        best_experiment = exp_id
        
        return best_experiment
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Log metrics for an experiment."""
        metrics_file = os.path.join(self.metrics_dir, f"{experiment_id}_metrics.json")
        
        # Load existing metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = {}
        
        # Update metrics
        existing_metrics.update(metrics)
        existing_metrics['updated_at'] = datetime.now().isoformat()
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    
    def archive_experiment(self, experiment_id: str):
        """Archive an experiment."""
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        archive_dir = os.path.join(self.experiments_dir, "archived", experiment_id)
        
        os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
        shutil.move(exp_dir, archive_dir)
        
        print(f" Archived experiment: {experiment_id}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    def _update_experiment_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        metadata_file = os.path.join(exp_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['status'] = status
            metadata['updated_at'] = datetime.now().isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _collect_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Collect experiment results."""
        results = {}
        
        # Load metrics
        metrics_file = os.path.join(self.metrics_dir, f"{experiment_id}_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results['metrics'] = json.load(f)
        
        # Check for model files
        model_patterns = [
            f"models/*{experiment_id}*",
            f"models/multi_agent*"
        ]
        
        results['models'] = []
        for pattern in model_patterns:
            import glob
            models = glob.glob(os.path.join(self.project_root, pattern))
            results['models'].extend(models)
        
        return results
    
    def _get_all_experiments(self) -> List[str]:
        """Get all experiment IDs."""
        if not os.path.exists(self.experiments_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, item)
            if os.path.isdir(exp_path) and item != "archived":
                experiments.append(item)
        
        return experiments

# Example usage functions
def create_hyperparameter_sweep():
    """Create multiple experiments for hyperparameter tuning."""
    tracker = DVCExperimentTracker()
    
    # Define parameter grid
    param_grid = {
        'learning_rates': [0.0001, 0.0003, 0.001],
        'batch_sizes': [32, 64, 128],
        'episodes': [500, 1000, 2000]
    }
    
    experiment_ids = []
    
    for lr in param_grid['learning_rates']:
        for batch_size in param_grid['batch_sizes']:
            for episodes in param_grid['episodes']:
                config_overrides = {
                    'training': {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'episodes': episodes
                    }
                }
                
                exp_name = f"sweep_lr{lr}_bs{batch_size}_ep{episodes}"
                exp_id = tracker.create_experiment(
                    name=exp_name,
                    description=f"Hyperparameter sweep: LR={lr}, BS={batch_size}, EP={episodes}",
                    tags=["hyperparameter_sweep", "dqn"],
                    config_overrides=config_overrides
                )
                experiment_ids.append(exp_id)
    
    return experiment_ids

def run_model_comparison():
    """Compare different model architectures."""
    tracker = DVCExperimentTracker()
    
    architectures = [
        {'hidden_layers': [128, 64], 'activation': 'relu'},
        {'hidden_layers': [256, 128, 64], 'activation': 'relu'},
        {'hidden_layers': [512, 256, 128], 'activation': 'tanh'},
        {'hidden_layers': [256, 128, 64], 'activation': 'leaky_relu'}
    ]
    
    experiment_ids = []
    
    for i, arch in enumerate(architectures):
        config_overrides = {'model': arch}
        
        exp_id = tracker.create_experiment(
            name=f"architecture_comparison_{i+1}",
            description=f"Architecture: {arch['hidden_layers']} with {arch['activation']}",
            tags=["architecture_comparison", "model_design"],
            config_overrides=config_overrides
        )
        experiment_ids.append(exp_id)
    
    return experiment_ids
