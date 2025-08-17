"""
Airflow DAG for Trading System Pipeline
Orchestrates DVC pipeline with proper scheduling and monitoring.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import os
import json
import yaml

# Default arguments for all tasks
default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
dag = DAG(
    'trading_system_pipeline',
    default_args=default_args,
    description='Complete trading system training and evaluation pipeline',
    schedule_interval='0 2 * * 1',  # Weekly on Monday at 2 AM
    max_active_runs=1,
    tags=['trading', 'ml', 'dvc']
)

# Configuration
PROJECT_ROOT = Variable.get("TRADING_PROJECT_ROOT", "/path/to/trading/project")
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
PYTHON_ENV = Variable.get("PYTHON_ENV", "python")

def check_market_data(**context):
    """Check if market data is available and recent."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Check if we have recent data for all symbols
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    
    for symbol in SYMBOLS:
        data_file = os.path.join(data_dir, f"{symbol}_processed.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file for {symbol} not found")
        
        # Check if data is recent (within last 7 days)
        df = pd.read_csv(data_file)
        if 'Date' in df.columns:
            last_date = pd.to_datetime(df['Date'].max())
            if datetime.now() - last_date > timedelta(days=7):
                raise ValueError(f"Data for {symbol} is stale (last: {last_date})")
    
    print("✅ All market data checks passed")
    return True

def validate_dvc_pipeline(**context):
    """Validate DVC pipeline configuration."""
    dvc_file = os.path.join(PROJECT_ROOT, "dvc.yaml")
    params_file = os.path.join(PROJECT_ROOT, "params.yaml")
    
    if not os.path.exists(dvc_file):
        raise FileNotFoundError("dvc.yaml not found")
    
    if not os.path.exists(params_file):
        raise FileNotFoundError("params.yaml not found")
    
    # Validate pipeline structure
    with open(dvc_file, 'r') as f:
        dvc_config = yaml.safe_load(f)
    
    required_stages = ['data_preparation', 'train_dqn', 'train_sac', 'train_multi_agent']
    for stage in required_stages:
        if stage not in dvc_config.get('stages', {}):
            raise ValueError(f"Required stage '{stage}' not found in DVC pipeline")
    
    print(" DVC pipeline validation passed")
    return True

def collect_training_metrics(**context):
    """Collect and validate training metrics."""
    metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
    
    if not os.path.exists(metrics_dir):
        raise FileNotFoundError("Metrics directory not found")
    
    # Collect metrics from all trained models
    metrics_summary = {}
    
    # Multi-agent metrics
    multi_agent_metrics_file = os.path.join(metrics_dir, "multi_agent_metrics.json")
    if os.path.exists(multi_agent_metrics_file):
        with open(multi_agent_metrics_file, 'r') as f:
            metrics_summary['multi_agent'] = json.load(f)
    
    # Individual agent metrics
    for symbol in SYMBOLS:
        for agent_type in ['dqn', 'sac']:
            metrics_file = os.path.join(metrics_dir, f"{agent_type}_{symbol}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_summary[f"{agent_type}_{symbol}"] = json.load(f)
    
    # Store metrics summary for downstream tasks
    context['task_instance'].xcom_push(key='metrics_summary', value=metrics_summary)
    
    print(f" Collected metrics for {len(metrics_summary)} models")
    return metrics_summary

def send_training_report(**context):
    """Send training completion report."""
    metrics_summary = context['task_instance'].xcom_pull(key='metrics_summary')
    
    # Generate report
    report = {
        'pipeline_run_date': context['ds'],
        'total_models_trained': len(metrics_summary),
        'models': list(metrics_summary.keys()),
        'status': 'completed'
    }
    
    # Add best performing model
    if 'multi_agent' in metrics_summary:
        ma_metrics = metrics_summary['multi_agent']
        if 'metrics' in ma_metrics:
            report['multi_agent_performance'] = {
                'total_return': ma_metrics['metrics'].get('portfolio_total_return', 'N/A'),
                'sharpe_ratio': ma_metrics['metrics'].get('portfolio_sharpe_ratio', 'N/A')
            }
    
    print("📊 Training Report:")
    print(json.dumps(report, indent=2))
    
    # Here you could send email, Slack notification, etc.
    return report

# Start of pipeline
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# Data validation tasks
data_validation_group = TaskGroup(
    group_id='data_validation',
    dag=dag
)

check_data_task = PythonOperator(
    task_id='check_market_data',
    python_callable=check_market_data,
    task_group=data_validation_group,
    dag=dag
)

validate_pipeline_task = PythonOperator(
    task_id='validate_dvc_pipeline',
    python_callable=validate_dvc_pipeline,
    task_group=data_validation_group,
    dag=dag
)

# Data preparation
data_prep_task = BashOperator(
    task_id='data_preparation',
    bash_command=f'cd {PROJECT_ROOT} && dvc repro data_preparation',
    dag=dag
)

# Individual agent training group
individual_training_group = TaskGroup(
    group_id='individual_agent_training',
    dag=dag
)

# DQN training tasks
dqn_tasks = []
for symbol in SYMBOLS:
    dqn_task = BashOperator(
        task_id=f'train_dqn_{symbol}',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro train_dqn:{symbol}',
        task_group=individual_training_group,
        dag=dag
    )
    dqn_tasks.append(dqn_task)

# SAC training tasks
sac_tasks = []
for symbol in SYMBOLS:
    sac_task = BashOperator(
        task_id=f'train_sac_{symbol}',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro train_sac:{symbol}',
        task_group=individual_training_group,
        dag=dag
    )
    sac_tasks.append(sac_task)

# Multi-agent training (depends on individual agents)
multi_agent_task = BashOperator(
    task_id='train_multi_agent',
    bash_command=f'cd {PROJECT_ROOT} && dvc repro train_multi_agent',
    dag=dag
)

# Model evaluation group
evaluation_group = TaskGroup(
    group_id='model_evaluation',
    dag=dag
)

# Evaluation tasks
eval_tasks = []
eval_models = [f'models/dqn_AAPL/', f'models/sac_AAPL/', 'models/multi_agent/']

for i, model_dir in enumerate(eval_models):
    eval_task = BashOperator(
        task_id=f'evaluate_model_{i}',
        bash_command=f'cd {PROJECT_ROOT} && dvc repro evaluate_models:{model_dir}',
        task_group=evaluation_group,
        dag=dag
    )
    eval_tasks.append(eval_task)

# Post-processing tasks
collect_metrics_task = PythonOperator(
    task_id='collect_training_metrics',
    python_callable=collect_training_metrics,
    dag=dag
)

send_report_task = PythonOperator(
    task_id='send_training_report',
    python_callable=send_training_report,
    dag=dag
)

# End of pipeline
end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> data_validation_group
data_validation_group >> data_prep_task
data_prep_task >> individual_training_group

# Individual training can run in parallel
for dqn_task, sac_task in zip(dqn_tasks, sac_tasks):
    # DQN and SAC for same symbol can run in parallel
    pass

# Multi-agent depends on all individual training
individual_training_group >> multi_agent_task

# Evaluation depends on multi-agent training
multi_agent_task >> evaluation_group

# Post-processing
evaluation_group >> collect_metrics_task
collect_metrics_task >> send_report_task
send_report_task >> end_task
