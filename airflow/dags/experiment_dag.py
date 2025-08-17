"""
Dynamic Airflow DAG for Trading Experiments
Creates dynamic tasks for hyperparameter sweeps and model comparisons.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import json
import itertools

# Default arguments
default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
dag = DAG(
    'trading_experiments',
    default_args=default_args,
    description='Dynamic experiment generation for trading system',
    schedule_interval=None,  # Triggered manually or by other DAGs
    max_active_runs=3,
    tags=['trading', 'experiments', 'hyperparameters']
)

PROJECT_ROOT = Variable.get("TRADING_PROJECT_ROOT", "/path/to/trading/project")

def generate_hyperparameter_experiments(**context):
    """Generate experiment configurations for hyperparameter sweep."""
    
    # Define parameter grid
    param_grid = {
        'learning_rates': [0.0001, 0.0003, 0.001],
        'batch_sizes': [32, 64, 128],
        'episodes': [500, 1000, 2000],
        'agent_types': [
            ['dqn', 'dqn', 'dqn', 'dqn'],
            ['sac', 'sac', 'sac', 'sac'],
            ['dqn', 'sac', 'dqn', 'sac']
        ]
    }
    
    experiments = []
    exp_id = 0
    
    for lr in param_grid['learning_rates']:
        for batch_size in param_grid['batch_sizes']:
            for episodes in param_grid['episodes']:
                for agent_types in param_grid['agent_types']:
                    
                    config_override = {
                        'training': {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'episodes': episodes
                        },
                        'sac': {
                            'learning_rate': lr,
                            'batch_size': batch_size
                        },
                        'multi_agent': {
                            'agent_types': agent_types
                        }
                    }
                    
                    agent_type_str = '_'.join(set(agent_types))
                    exp_name = f"sweep_lr{lr}_bs{batch_size}_ep{episodes}_{agent_type_str}"
                    
                    experiments.append({
                        'id': exp_id,
                        'name': exp_name,
                        'config': config_override,
                        'tags': ['hyperparameter_sweep', agent_type_str]
                    })
                    exp_id += 1
    
    # Store experiments for downstream tasks
    context['task_instance'].xcom_push(key='experiments', value=experiments)
    print(f"Generated {len(experiments)} experiments")
    
    return experiments

def run_experiment(experiment_config, **context):
    """Run a single experiment with given configuration."""
    import tempfile
    import yaml
    import os
    
    exp_name = experiment_config['name']
    config_override = experiment_config['config']
    
    print(f"Running experiment: {exp_name}")
    
    # Create temporary parameter file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        # Load base params
        base_params_file = os.path.join(PROJECT_ROOT, 'params.yaml')
        with open(base_params_file, 'r') as base_f:
            base_params = yaml.safe_load(base_f)
        
        # Apply overrides
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(base_params, config_override)
        
        # Add experiment metadata
        base_params['experiment'] = {
            'name': exp_name,
            'id': experiment_config['id'],
            'tags': experiment_config['tags'],
            'created_at': datetime.now().isoformat()
        }
        
        yaml.dump(base_params, f)
        temp_params_file = f.name
    
    try:
        # Copy temp params to main params file
        import shutil
        main_params_file = os.path.join(PROJECT_ROOT, 'params.yaml')
        shutil.copy2(temp_params_file, main_params_file)
        
        # Run DVC pipeline
        import subprocess
        result = subprocess.run(
            ['dvc', 'repro', 'train_multi_agent'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"DVC pipeline failed: {result.stderr}")
        
        print(f"✅ Experiment {exp_name} completed successfully")
        
        # Collect metrics
        metrics_file = os.path.join(PROJECT_ROOT, 'metrics', 'multi_agent_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Store metrics with experiment info
            experiment_result = {
                'experiment': experiment_config,
                'metrics': metrics,
                'status': 'completed'
            }
            
            context['task_instance'].xcom_push(
                key=f'experiment_{experiment_config["id"]}_result', 
                value=experiment_result
            )
            
            return experiment_result
        
    finally:
        # Cleanup temp file
        os.unlink(temp_params_file)
    
    return {'status': 'completed', 'experiment': experiment_config}

def collect_experiment_results(**context):
    """Collect results from all experiments."""
    experiments = context['task_instance'].xcom_pull(key='experiments')
    
    results = []
    for exp in experiments:
        exp_result = context['task_instance'].xcom_pull(
            key=f'experiment_{exp["id"]}_result'
        )
        if exp_result:
            results.append(exp_result)
    
    # Analyze results
    if results:
        # Find best experiment by portfolio return
        best_exp = max(
            results, 
            key=lambda x: x.get('metrics', {}).get('metrics', {}).get('portfolio_total_return', -float('inf'))
        )
        
        print(f"🏆 Best experiment: {best_exp['experiment']['name']}")
        print(f"   Portfolio return: {best_exp.get('metrics', {}).get('metrics', {}).get('portfolio_total_return', 'N/A')}")
        
        # Store summary
        summary = {
            'total_experiments': len(results),
            'completed_experiments': len([r for r in results if r['status'] == 'completed']),
            'best_experiment': best_exp['experiment'],
            'best_metrics': best_exp.get('metrics', {})
        }
        
        context['task_instance'].xcom_push(key='experiment_summary', value=summary)
        
        return summary
    
    return {'total_experiments': 0}

# DAG tasks
start_experiments = DummyOperator(
    task_id='start_experiments',
    dag=dag
)

generate_experiments_task = PythonOperator(
    task_id='generate_experiments',
    python_callable=generate_hyperparameter_experiments,
    dag=dag
)

# Dynamic task creation for experiments
def create_experiment_tasks():
    """Create tasks dynamically based on experiment configuration."""
    
    # This would be called during DAG parsing
    # For now, we'll create a fixed number of parallel experiment slots
    experiment_tasks = []
    
    for i in range(10):  # Max 10 parallel experiments
        task = PythonOperator(
            task_id=f'run_experiment_{i}',
            python_callable=lambda **context: run_single_experiment_slot(i, **context),
            dag=dag
        )
        experiment_tasks.append(task)
    
    return experiment_tasks

def run_single_experiment_slot(slot_id, **context):
    """Run experiment assigned to this slot."""
    experiments = context['task_instance'].xcom_pull(key='experiments')
    
    if not experiments or slot_id >= len(experiments):
        print(f"No experiment assigned to slot {slot_id}")
        return {'status': 'skipped'}
    
    experiment = experiments[slot_id]
    return run_experiment(experiment, **context)

# Create experiment tasks
experiment_tasks = create_experiment_tasks()

collect_results_task = PythonOperator(
    task_id='collect_experiment_results',
    python_callable=collect_experiment_results,
    dag=dag
)

end_experiments = DummyOperator(
    task_id='end_experiments',
    dag=dag
)

# Define dependencies
start_experiments >> generate_experiments_task

for task in experiment_tasks:
    generate_experiments_task >> task
    task >> collect_results_task

collect_results_task >> end_experiments
