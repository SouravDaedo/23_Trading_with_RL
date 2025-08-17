"""
Monitoring and Alerting DAG for Trading System
Monitors model performance, data quality, and system health.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import json
import pandas as pd
import os

default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'catchup': False
}

dag = DAG(
    'trading_system_monitoring',
    default_args=default_args,
    description='Monitor trading system performance and health',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    max_active_runs=1,
    tags=['monitoring', 'alerts', 'trading']
)

PROJECT_ROOT = Variable.get("TRADING_PROJECT_ROOT", "/path/to/trading/project")
ALERT_EMAIL = Variable.get("ALERT_EMAIL", "admin@trading.com")
PERFORMANCE_THRESHOLD = float(Variable.get("PERFORMANCE_THRESHOLD", "0.05"))  # 5% return threshold

def check_model_performance(**context):
    """Check if models are performing within expected ranges."""
    metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
    alerts = []
    
    # Check multi-agent performance
    multi_agent_file = os.path.join(metrics_dir, "multi_agent_metrics.json")
    if os.path.exists(multi_agent_file):
        with open(multi_agent_file, 'r') as f:
            metrics = json.load(f)
        
        portfolio_return = metrics.get('metrics', {}).get('portfolio_total_return', 0)
        sharpe_ratio = metrics.get('metrics', {}).get('portfolio_sharpe_ratio', 0)
        max_drawdown = metrics.get('metrics', {}).get('portfolio_max_drawdown', 0)
        
        # Performance checks
        if portfolio_return < -0.1:  # -10% return
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'message': f"Multi-agent portfolio return is {portfolio_return:.2%} (below -10%)",
                'metric': 'portfolio_return',
                'value': portfolio_return
            })
        
        if sharpe_ratio < 0:
            alerts.append({
                'type': 'performance',
                'severity': 'medium',
                'message': f"Multi-agent Sharpe ratio is negative: {sharpe_ratio:.3f}",
                'metric': 'sharpe_ratio',
                'value': sharpe_ratio
            })
        
        if abs(max_drawdown) > 0.2:  # 20% drawdown
            alerts.append({
                'type': 'risk',
                'severity': 'high',
                'message': f"Multi-agent max drawdown is {max_drawdown:.2%} (above 20%)",
                'metric': 'max_drawdown',
                'value': max_drawdown
            })
    
    # Check individual agent performance
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    for symbol in symbols:
        for agent_type in ['dqn', 'sac']:
            agent_file = os.path.join(metrics_dir, f"{agent_type}_{symbol}_metrics.json")
            if os.path.exists(agent_file):
                with open(agent_file, 'r') as f:
                    agent_metrics = json.load(f)
                
                agent_return = agent_metrics.get('metrics', {}).get('total_return', 0)
                if agent_return < -0.15:  # -15% for individual agents
                    alerts.append({
                        'type': 'agent_performance',
                        'severity': 'medium',
                        'message': f"{agent_type.upper()} agent for {symbol} return: {agent_return:.2%}",
                        'agent': f"{agent_type}_{symbol}",
                        'value': agent_return
                    })
    
    # Store alerts for downstream tasks
    context['task_instance'].xcom_push(key='performance_alerts', value=alerts)
    
    if alerts:
        print(f"⚠️ Found {len(alerts)} performance alerts")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    else:
        print("✅ All models performing within expected ranges")
    
    return alerts

def check_data_quality(**context):
    """Check data quality and freshness."""
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    alerts = []
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        data_file = os.path.join(data_dir, f"{symbol}_processed.csv")
        
        if not os.path.exists(data_file):
            alerts.append({
                'type': 'data_missing',
                'severity': 'high',
                'message': f"Data file missing for {symbol}",
                'symbol': symbol
            })
            continue
        
        try:
            df = pd.read_csv(data_file)
            
            # Check data freshness
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                last_date = df['Date'].max()
                days_old = (datetime.now() - last_date).days
                
                if days_old > 7:
                    alerts.append({
                        'type': 'data_stale',
                        'severity': 'medium',
                        'message': f"Data for {symbol} is {days_old} days old (last: {last_date.date()})",
                        'symbol': symbol,
                        'days_old': days_old
                    })
            
            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_pct > 0.05:  # 5% missing data
                alerts.append({
                    'type': 'data_quality',
                    'severity': 'medium',
                    'message': f"High missing data percentage for {symbol}: {missing_pct:.2%}",
                    'symbol': symbol,
                    'missing_pct': missing_pct
                })
            
            # Check for data anomalies (e.g., extreme price movements)
            if 'Close' in df.columns:
                price_changes = df['Close'].pct_change().abs()
                extreme_changes = (price_changes > 0.2).sum()  # 20% daily change
                
                if extreme_changes > len(df) * 0.01:  # More than 1% of days
                    alerts.append({
                        'type': 'data_anomaly',
                        'severity': 'low',
                        'message': f"Unusual price movements detected for {symbol}: {extreme_changes} extreme days",
                        'symbol': symbol,
                        'extreme_days': extreme_changes
                    })
        
        except Exception as e:
            alerts.append({
                'type': 'data_error',
                'severity': 'high',
                'message': f"Error reading data for {symbol}: {str(e)}",
                'symbol': symbol
            })
    
    context['task_instance'].xcom_push(key='data_alerts', value=alerts)
    
    if alerts:
        print(f"⚠️ Found {len(alerts)} data quality alerts")
    else:
        print("✅ Data quality checks passed")
    
    return alerts

def check_system_health(**context):
    """Check system health and resource usage."""
    alerts = []
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_pct = free / total
    
    if free_pct < 0.1:  # Less than 10% free space
        alerts.append({
            'type': 'disk_space',
            'severity': 'high',
            'message': f"Low disk space: {free_pct:.1%} free ({free // (1024**3):.1f} GB)",
            'free_space_gb': free // (1024**3),
            'free_pct': free_pct
        })
    
    # Check model file sizes
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if os.path.exists(models_dir):
        total_model_size = 0
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_model_size += os.path.getsize(file_path)
        
        total_model_size_gb = total_model_size / (1024**3)
        if total_model_size_gb > 10:  # More than 10GB of models
            alerts.append({
                'type': 'model_storage',
                'severity': 'medium',
                'message': f"Large model storage usage: {total_model_size_gb:.1f} GB",
                'model_size_gb': total_model_size_gb
            })
    
    # Check for failed training runs
    metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
    if os.path.exists(metrics_dir):
        recent_metrics = []
        for file in os.listdir(metrics_dir):
            if file.endswith('_metrics.json'):
                file_path = os.path.join(metrics_dir, file)
                file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).days
                if file_age <= 7:  # Files from last week
                    recent_metrics.append(file)
        
        if len(recent_metrics) == 0:
            alerts.append({
                'type': 'training_status',
                'severity': 'medium',
                'message': "No recent training metrics found (last 7 days)",
                'days_since_training': 7
            })
    
    context['task_instance'].xcom_push(key='system_alerts', value=alerts)
    
    if alerts:
        print(f"⚠️ Found {len(alerts)} system health alerts")
    else:
        print("✅ System health checks passed")
    
    return alerts

def generate_alert_report(**context):
    """Generate comprehensive alert report."""
    performance_alerts = context['task_instance'].xcom_pull(key='performance_alerts') or []
    data_alerts = context['task_instance'].xcom_pull(key='data_alerts') or []
    system_alerts = context['task_instance'].xcom_pull(key='system_alerts') or []
    
    all_alerts = performance_alerts + data_alerts + system_alerts
    
    if not all_alerts:
        print("✅ No alerts to report - system healthy")
        return {'status': 'healthy', 'total_alerts': 0}
    
    # Categorize alerts by severity
    high_alerts = [a for a in all_alerts if a.get('severity') == 'high']
    medium_alerts = [a for a in all_alerts if a.get('severity') == 'medium']
    low_alerts = [a for a in all_alerts if a.get('severity') == 'low']
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_alerts': len(all_alerts),
        'high_severity': len(high_alerts),
        'medium_severity': len(medium_alerts),
        'low_severity': len(low_alerts),
        'alerts': {
            'high': high_alerts,
            'medium': medium_alerts,
            'low': low_alerts
        }
    }
    
    # Generate summary message
    summary_lines = [
        f"🚨 Trading System Alert Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total Alerts: {len(all_alerts)} (High: {len(high_alerts)}, Medium: {len(medium_alerts)}, Low: {len(low_alerts)})",
        ""
    ]
    
    if high_alerts:
        summary_lines.append("🔴 HIGH SEVERITY ALERTS:")
        for alert in high_alerts:
            summary_lines.append(f"  • {alert['message']}")
        summary_lines.append("")
    
    if medium_alerts:
        summary_lines.append("🟡 MEDIUM SEVERITY ALERTS:")
        for alert in medium_alerts:
            summary_lines.append(f"  • {alert['message']}")
        summary_lines.append("")
    
    if low_alerts:
        summary_lines.append("🟢 LOW SEVERITY ALERTS:")
        for alert in low_alerts:
            summary_lines.append(f"  • {alert['message']}")
    
    report['summary'] = '\n'.join(summary_lines)
    
    # Store report for email task
    context['task_instance'].xcom_push(key='alert_report', value=report)
    
    print(report['summary'])
    
    return report

def send_alert_email(**context):
    """Send alert email if there are high or medium severity alerts."""
    report = context['task_instance'].xcom_pull(key='alert_report')
    
    if not report or (report['high_severity'] == 0 and report['medium_severity'] == 0):
        print("No critical alerts - skipping email")
        return "skipped"
    
    # This would integrate with your email system
    print(f"📧 Would send alert email to {ALERT_EMAIL}")
    print(f"Subject: Trading System Alerts - {report['high_severity']} High, {report['medium_severity']} Medium")
    
    return "sent"

# Define tasks
start_monitoring = DummyOperator(
    task_id='start_monitoring',
    dag=dag
)

check_performance_task = PythonOperator(
    task_id='check_model_performance',
    python_callable=check_model_performance,
    dag=dag
)

check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

check_system_task = PythonOperator(
    task_id='check_system_health',
    python_callable=check_system_health,
    dag=dag
)

generate_report_task = PythonOperator(
    task_id='generate_alert_report',
    python_callable=generate_alert_report,
    dag=dag
)

send_email_task = PythonOperator(
    task_id='send_alert_email',
    python_callable=send_alert_email,
    dag=dag
)

end_monitoring = DummyOperator(
    task_id='end_monitoring',
    dag=dag
)

# Define dependencies
start_monitoring >> [check_performance_task, check_data_task, check_system_task]
[check_performance_task, check_data_task, check_system_task] >> generate_report_task
generate_report_task >> send_email_task >> end_monitoring
