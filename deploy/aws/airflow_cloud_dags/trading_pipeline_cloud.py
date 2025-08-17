"""
Cloud-optimized Airflow DAG for Trading System
Uses AWS ECS for training workloads with MWAA.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import json

# Default arguments
default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'catchup': False
}

# DAG definition
dag = DAG(
    'trading_system_cloud_pipeline',
    default_args=default_args,
    description='Cloud-based trading system training pipeline',
    schedule_interval='0 2 * * 1',  # Weekly on Monday at 2 AM
    max_active_runs=1,
    tags=['trading', 'cloud', 'ecs', 'ml']
)

# Configuration from Airflow Variables
AWS_REGION = Variable.get("AWS_REGION", "us-east-1")
ECS_CLUSTER = Variable.get("ECS_CLUSTER", "trading-system-cluster")
TASK_DEFINITION = Variable.get("TASK_DEFINITION", "trading-system-training")
SUBNET_IDS = Variable.get("SUBNET_IDS", "").split(",")
SECURITY_GROUP_IDS = Variable.get("SECURITY_GROUP_IDS", "").split(",")
S3_BUCKET = Variable.get("S3_BUCKET", "trading-system-data")

def prepare_training_config(**context):
    """Prepare configuration for cloud training."""
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'training': {
            'episodes': 1000,
            'batch_size': 64,
            'learning_rate': 0.0003
        },
        'multi_agent': {
            'agent_types': ['dqn', 'sac', 'dqn', 'sac']
        },
        'cloud': {
            's3_bucket': S3_BUCKET,
            'run_id': context['run_id'],
            'execution_date': context['ds']
        }
    }
    
    # Store config for downstream tasks
    context['task_instance'].xcom_push(key='training_config', value=config)
    return config

def create_ecs_task_config(task_name: str, command: list, **context):
    """Create ECS task configuration."""
    config = context['task_instance'].xcom_pull(key='training_config')
    
    return {
        'taskDefinitionArn': TASK_DEFINITION,
        'cluster': ECS_CLUSTER,
        'launchType': 'FARGATE',
        'networkConfiguration': {
            'awsvpcConfiguration': {
                'subnets': SUBNET_IDS,
                'securityGroups': SECURITY_GROUP_IDS,
                'assignPublicIp': 'DISABLED'
            }
        },
        'overrides': {
            'containerOverrides': [
                {
                    'name': 'trading-trainer',
                    'command': command,
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                        {'name': 'RUN_ID', 'value': context['run_id']},
                        {'name': 'TASK_NAME', 'value': task_name},
                        {'name': 'CONFIG_JSON', 'value': json.dumps(config)}
                    ]
                }
            ]
        }
    }

# Start of pipeline
start_task = DummyOperator(
    task_id='start_cloud_pipeline',
    dag=dag
)

# Prepare configuration
prepare_config_task = PythonOperator(
    task_id='prepare_training_config',
    python_callable=prepare_training_config,
    dag=dag
)

# Upload configuration to S3
upload_config_task = S3CreateObjectOperator(
    task_id='upload_config_to_s3',
    s3_bucket=S3_BUCKET,
    s3_key='configs/{{ run_id }}/params.yaml',
    data="{{ task_instance.xcom_pull(key='training_config') | tojson }}",
    dag=dag
)

# Data preparation task
data_prep_task = EcsRunTaskOperator(
    task_id='data_preparation_cloud',
    task_definition=TASK_DEFINITION,
    cluster=ECS_CLUSTER,
    launch_type='FARGATE',
    network_configuration={
        'awsvpcConfiguration': {
            'subnets': SUBNET_IDS,
            'securityGroups': SECURITY_GROUP_IDS,
            'assignPublicIp': 'DISABLED'
        }
    },
    overrides={
        'containerOverrides': [
            {
                'name': 'trading-trainer',
                'command': ['python', '-c', 'from src.data.data_loader import DataLoader; dl = DataLoader(); [dl.load_data(s) for s in ["AAPL", "GOOGL", "MSFT", "TSLA"]]'],
                'environment': [
                    {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                    {'name': 'TASK_NAME', 'value': 'data_preparation'}
                ]
            }
        ]
    },
    dag=dag
)

# Individual agent training group
individual_training_group = TaskGroup(
    group_id='individual_agent_training_cloud',
    dag=dag
)

# DQN training tasks (parallel)
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
dqn_tasks = []

for symbol in symbols:
    dqn_task = EcsRunTaskOperator(
        task_id=f'train_dqn_{symbol}_cloud',
        task_definition=TASK_DEFINITION,
        cluster=ECS_CLUSTER,
        launch_type='FARGATE',
        network_configuration={
            'awsvpcConfiguration': {
                'subnets': SUBNET_IDS,
                'securityGroups': SECURITY_GROUP_IDS,
                'assignPublicIp': 'DISABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'trading-trainer',
                    'command': ['python', 'train_agent_unified.py', '--agent', 'dqn', '--symbol', symbol],
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                        {'name': 'SYMBOL', 'value': symbol},
                        {'name': 'AGENT_TYPE', 'value': 'dqn'}
                    ]
                }
            ]
        },
        task_group=individual_training_group,
        dag=dag
    )
    dqn_tasks.append(dqn_task)

# SAC training tasks (parallel)
sac_tasks = []

for symbol in symbols:
    sac_task = EcsRunTaskOperator(
        task_id=f'train_sac_{symbol}_cloud',
        task_definition=TASK_DEFINITION,
        cluster=ECS_CLUSTER,
        launch_type='FARGATE',
        network_configuration={
            'awsvpcConfiguration': {
                'subnets': SUBNET_IDS,
                'securityGroups': SECURITY_GROUP_IDS,
                'assignPublicIp': 'DISABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'trading-trainer',
                    'command': ['python', 'train_agent_unified.py', '--agent', 'sac', '--symbol', symbol],
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                        {'name': 'SYMBOL', 'value': symbol},
                        {'name': 'AGENT_TYPE', 'value': 'sac'}
                    ]
                }
            ]
        },
        task_group=individual_training_group,
        dag=dag
    )
    sac_tasks.append(sac_task)

# Multi-agent training (high-memory task)
multi_agent_task = EcsRunTaskOperator(
    task_id='train_multi_agent_cloud',
    task_definition=TASK_DEFINITION,
    cluster=ECS_CLUSTER,
    launch_type='FARGATE',
    network_configuration={
        'awsvpcConfiguration': {
            'subnets': SUBNET_IDS,
            'securityGroups': SECURITY_GROUP_IDS,
            'assignPublicIp': 'DISABLED'
        }
    },
    overrides={
        'containerOverrides': [
            {
                'name': 'trading-trainer',
                'command': ['python', 'train_multi_agent.py'],
                'environment': [
                    {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                    {'name': 'TASK_NAME', 'value': 'multi_agent_training'}
                ],
                'memory': 16384,  # 16GB for multi-agent training
                'cpu': 4096      # 4 vCPUs
            }
        ]
    },
    dag=dag
)

# Model evaluation group
evaluation_group = TaskGroup(
    group_id='model_evaluation_cloud',
    dag=dag
)

# Evaluation tasks
eval_models = ['dqn_AAPL', 'sac_AAPL', 'multi_agent']
eval_tasks = []

for model in eval_models:
    eval_task = EcsRunTaskOperator(
        task_id=f'evaluate_{model}_cloud',
        task_definition=TASK_DEFINITION,
        cluster=ECS_CLUSTER,
        launch_type='FARGATE',
        network_configuration={
            'awsvpcConfiguration': {
                'subnets': SUBNET_IDS,
                'securityGroups': SECURITY_GROUP_IDS,
                'assignPublicIp': 'DISABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'trading-trainer',
                    'command': ['python', 'evaluate_agent.py', '--model-dir', f'models/{model}/'],
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                        {'name': 'MODEL_NAME', 'value': model}
                    ]
                }
            ]
        },
        task_group=evaluation_group,
        dag=dag
    )
    eval_tasks.append(eval_task)

def collect_cloud_metrics(**context):
    """Collect metrics from S3 after cloud training."""
    import boto3
    
    s3 = boto3.client('s3')
    run_id = context['run_id']
    
    # List metrics files from S3
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=f'metrics/{run_id}/'
    )
    
    metrics_summary = {}
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('_metrics.json'):
                # Download and parse metrics
                response = s3.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                metrics_data = json.loads(response['Body'].read())
                
                # Extract model name from key
                model_name = obj['Key'].split('/')[-1].replace('_metrics.json', '')
                metrics_summary[model_name] = metrics_data
    
    context['task_instance'].xcom_push(key='cloud_metrics_summary', value=metrics_summary)
    print(f"Collected metrics for {len(metrics_summary)} models from S3")
    
    return metrics_summary

def send_cloud_training_report(**context):
    """Send training completion report for cloud execution."""
    metrics_summary = context['task_instance'].xcom_pull(key='cloud_metrics_summary')
    
    report = {
        'pipeline_run_date': context['ds'],
        'run_id': context['run_id'],
        'execution_mode': 'cloud',
        'total_models_trained': len(metrics_summary),
        'models': list(metrics_summary.keys()),
        'status': 'completed',
        's3_bucket': S3_BUCKET
    }
    
    # Add performance summary
    if 'multi_agent' in metrics_summary:
        ma_metrics = metrics_summary['multi_agent']
        if 'metrics' in ma_metrics:
            report['multi_agent_performance'] = {
                'total_return': ma_metrics['metrics'].get('portfolio_total_return', 'N/A'),
                'sharpe_ratio': ma_metrics['metrics'].get('portfolio_sharpe_ratio', 'N/A')
            }
    
    print("Cloud Training Report:")
    print(json.dumps(report, indent=2))
    
    # Store report in S3
    import boto3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f'reports/{context["run_id"]}/training_report.json',
        Body=json.dumps(report, indent=2)
    )
    
    return report

# Post-processing tasks
collect_metrics_task = PythonOperator(
    task_id='collect_cloud_metrics',
    python_callable=collect_cloud_metrics,
    dag=dag
)

send_report_task = PythonOperator(
    task_id='send_cloud_training_report',
    python_callable=send_cloud_training_report,
    dag=dag
)

# End of pipeline
end_task = DummyOperator(
    task_id='end_cloud_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> prepare_config_task >> upload_config_task >> data_prep_task
data_prep_task >> individual_training_group
individual_training_group >> multi_agent_task
multi_agent_task >> evaluation_group
evaluation_group >> collect_metrics_task >> send_report_task >> end_task
