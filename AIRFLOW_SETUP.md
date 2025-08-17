# Airflow Orchestration Setup Guide

This guide shows how to orchestrate your DVC trading pipeline using Apache Airflow for better scheduling, monitoring, and dependency management.

## 🚀 Quick Setup

### 1. Install Airflow
```bash
# Install Apache Airflow
pip install apache-airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 2. Configure Airflow
```bash
# Set Airflow home (optional)
export AIRFLOW_HOME=/path/to/your/airflow

# Copy configuration
cp airflow/airflow.cfg $AIRFLOW_HOME/airflow.cfg

# Update paths in airflow.cfg:
# - dags_folder = /path/to/trading/project/airflow/dags
# - base_log_folder = /path/to/airflow/logs
```

### 3. Set Airflow Variables
```bash
# Set project root path
airflow variables set TRADING_PROJECT_ROOT "/path/to/trading/project"

# Set Python environment
airflow variables set PYTHON_ENV "python"

# Set alert email
airflow variables set ALERT_EMAIL "your-email@example.com"

# Set performance threshold
airflow variables set PERFORMANCE_THRESHOLD "0.05"
```

### 4. Start Airflow
```bash
# Start scheduler (in one terminal)
airflow scheduler

# Start webserver (in another terminal)
airflow webserver --port 8080
```

## 📊 Available DAGs

### 1. Main Trading Pipeline (`trading_pipeline_dag.py`)
**Schedule**: Weekly on Monday at 2 AM  
**Purpose**: Complete training and evaluation pipeline

**Stages**:
- **Data Validation**: Check market data quality and DVC pipeline
- **Data Preparation**: Load and preprocess market data
- **Individual Training**: Parallel DQN and SAC training for each symbol
- **Multi-Agent Training**: Coordinated multi-agent system training
- **Model Evaluation**: Test all trained models
- **Reporting**: Collect metrics and send completion report

### 2. Experiment Pipeline (`experiment_dag.py`)
**Schedule**: Manual trigger  
**Purpose**: Dynamic hyperparameter sweeps and model comparisons

**Features**:
- **Dynamic task generation**: Creates experiments based on parameter grids
- **Parallel execution**: Runs up to 10 experiments simultaneously
- **Result comparison**: Automatically finds best performing configuration
- **Flexible parameters**: Learning rates, batch sizes, episodes, agent types

### 3. Monitoring Pipeline (`monitoring_dag.py`)
**Schedule**: Every 6 hours  
**Purpose**: System health and performance monitoring

**Checks**:
- **Model Performance**: Portfolio returns, Sharpe ratios, drawdowns
- **Data Quality**: Freshness, missing values, anomalies
- **System Health**: Disk space, model storage, training status
- **Alerting**: Email notifications for critical issues

## 🎯 Usage Examples

### Running the Main Pipeline
```bash
# Trigger manually
airflow dags trigger trading_system_pipeline

# Check status
airflow dags state trading_system_pipeline 2024-01-15

# View logs
airflow tasks log trading_system_pipeline train_multi_agent 2024-01-15
```

### Running Experiments
```bash
# Trigger experiment sweep
airflow dags trigger trading_experiments

# Monitor progress
airflow dags list-runs trading_experiments

# Check experiment results
airflow tasks log trading_experiments collect_experiment_results 2024-01-15
```

### Monitoring System
```bash
# Check monitoring status
airflow dags state trading_system_monitoring 2024-01-15

# View alerts
airflow tasks log trading_system_monitoring generate_alert_report 2024-01-15
```

## 🔧 DAG Configuration

### Main Pipeline Tasks
```python
# Data validation group
- check_market_data: Verify data availability and freshness
- validate_dvc_pipeline: Check DVC configuration

# Training tasks (parallel)
- train_dqn_AAPL, train_dqn_GOOGL, etc.: Individual DQN agents
- train_sac_AAPL, train_sac_GOOGL, etc.: Individual SAC agents

# Multi-agent training
- train_multi_agent: Coordinated system training

# Evaluation and reporting
- evaluate_model_0, evaluate_model_1, etc.: Model testing
- collect_training_metrics: Gather performance data
- send_training_report: Generate completion report
```

### Experiment Pipeline Features
```python
# Dynamic experiment generation
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

# Automatic best model selection
best_exp = max(results, key=lambda x: x['portfolio_total_return'])
```

### Monitoring Thresholds
```python
# Performance alerts
portfolio_return < -0.1  # -10% return threshold
sharpe_ratio < 0         # Negative Sharpe ratio
max_drawdown > 0.2       # 20% drawdown threshold

# Data quality alerts
days_old > 7             # Stale data threshold
missing_pct > 0.05       # 5% missing data threshold
extreme_changes > 1%     # Unusual price movements

# System health alerts
free_space < 10%         # Disk space threshold
model_size > 10GB        # Model storage threshold
no_training > 7_days     # Training activity threshold
```

## 📈 Monitoring and Alerting

### Performance Monitoring
- **Portfolio metrics**: Total return, Sharpe ratio, max drawdown
- **Agent performance**: Individual agent returns and win rates
- **Training progress**: Convergence, stability, loss trends

### Data Quality Checks
- **Freshness**: Data age and update frequency
- **Completeness**: Missing values and data gaps
- **Anomalies**: Unusual price movements or patterns

### System Health
- **Resource usage**: Disk space, memory, CPU
- **Storage management**: Model file sizes and cleanup
- **Pipeline status**: Training success rates and failures

### Alert Levels
- **🔴 High**: Immediate attention required (performance < -10%, missing data)
- **🟡 Medium**: Monitor closely (negative Sharpe, stale data)
- **🟢 Low**: Informational (minor anomalies, storage warnings)

## 🔄 Integration with DVC

### DVC Command Integration
```python
# Run specific DVC stages
BashOperator(
    task_id='train_multi_agent',
    bash_command='cd {PROJECT_ROOT} && dvc repro train_multi_agent'
)

# Run with parameter overrides
BashOperator(
    bash_command='cd {PROJECT_ROOT} && dvc exp run -S training.learning_rate=0.001'
)
```

### Parameter Management
```python
# Load DVC parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Apply experiment overrides
deep_update(params, config_overrides)

# Save updated parameters
with open('params.yaml', 'w') as f:
    yaml.dump(params, f)
```

### Metrics Collection
```python
# Collect DVC metrics
metrics_file = 'metrics/multi_agent_metrics.json'
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

# Store in Airflow XCom
context['task_instance'].xcom_push(key='metrics', value=metrics)
```

## 🛠️ Advanced Configuration

### Parallel Execution
```python
# Configure parallelism in airflow.cfg
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 16

# Use task groups for organization
with TaskGroup('individual_training') as training_group:
    dqn_tasks = [create_dqn_task(symbol) for symbol in SYMBOLS]
    sac_tasks = [create_sac_task(symbol) for symbol in SYMBOLS]
```

### Resource Management
```python
# Set resource requirements
task = BashOperator(
    task_id='train_multi_agent',
    bash_command='...',
    pool='gpu_pool',  # Use GPU pool
    queue='high_memory'  # High memory queue
)
```

### Error Handling
```python
# Retry configuration
default_args = {
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False
}

# Custom failure handling
def handle_failure(context):
    # Custom failure logic
    send_slack_notification(context)

task = PythonOperator(
    python_callable=my_function,
    on_failure_callback=handle_failure
)
```

## 🚨 Troubleshooting

### Common Issues

1. **DAG not appearing**
   ```bash
   # Check DAG syntax
   python airflow/dags/trading_pipeline_dag.py
   
   # Refresh DAGs
   airflow dags list-import-errors
   ```

2. **Task failures**
   ```bash
   # Check task logs
   airflow tasks log trading_system_pipeline train_multi_agent 2024-01-15
   
   # Clear failed tasks
   airflow tasks clear trading_system_pipeline --start_date 2024-01-15
   ```

3. **DVC integration issues**
   ```bash
   # Verify DVC status
   cd /path/to/trading/project && dvc status
   
   # Check DVC configuration
   dvc config list
   ```

### Performance Optimization

1. **Increase parallelism**
   ```ini
   # In airflow.cfg
   parallelism = 64
   dag_concurrency = 32
   ```

2. **Use connection pooling**
   ```python
   # For database connections
   sql_alchemy_pool_size = 10
   sql_alchemy_max_overflow = 20
   ```

3. **Optimize task scheduling**
   ```python
   # Use depends_on_past=False for parallel execution
   # Set appropriate pool sizes for resource management
   ```

This Airflow setup provides enterprise-grade orchestration for your trading system with comprehensive monitoring, alerting, and experiment management capabilities.
