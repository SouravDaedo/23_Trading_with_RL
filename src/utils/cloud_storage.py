"""
Cloud Storage Integration for Trading System
Handles S3 operations for models, data, and metrics.
"""

import os
import json
import boto3
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import pickle
import yaml

class CloudStorageManager:
    """Manages cloud storage operations for the trading system."""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """
        Initialize cloud storage manager.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)
        self.bucket = self.s3_resource.Bucket(bucket_name)
        
    def upload_data(self, local_path: str, s3_key: str) -> bool:
        """Upload data file to S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading {local_path}: {e}")
            return False
    
    def download_data(self, s3_key: str, local_path: str) -> bool:
        """Download data file from S3."""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            print(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_model(self, model_dir: str, run_id: str, model_name: str) -> bool:
        """Upload trained model directory to S3."""
        try:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, model_dir)
                    s3_key = f"models/{run_id}/{model_name}/{relative_path}"
                    
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            
            print(f"Uploaded model {model_name} to s3://{self.bucket_name}/models/{run_id}/{model_name}/")
            return True
        except Exception as e:
            print(f"Error uploading model {model_name}: {e}")
            return False
    
    def download_model(self, run_id: str, model_name: str, local_dir: str) -> bool:
        """Download trained model from S3."""
        try:
            prefix = f"models/{run_id}/{model_name}/"
            
            for obj in self.bucket.objects.filter(Prefix=prefix):
                relative_path = obj.key[len(prefix):]
                local_file = os.path.join(local_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                self.bucket.download_file(obj.key, local_file)
            
            print(f"Downloaded model {model_name} to {local_dir}")
            return True
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            return False
    
    def upload_metrics(self, metrics: Dict[str, Any], run_id: str, model_name: str) -> bool:
        """Upload metrics to S3."""
        try:
            s3_key = f"metrics/{run_id}/{model_name}_metrics.json"
            
            # Add timestamp
            metrics['uploaded_at'] = datetime.now().isoformat()
            metrics['run_id'] = run_id
            metrics['model_name'] = model_name
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(metrics, indent=2, default=str),
                ContentType='application/json'
            )
            
            print(f"Uploaded metrics for {model_name} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading metrics for {model_name}: {e}")
            return False
    
    def download_metrics(self, run_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Download metrics from S3."""
        try:
            s3_key = f"metrics/{run_id}/{model_name}_metrics.json"
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            metrics = json.loads(response['Body'].read())
            
            return metrics
        except Exception as e:
            print(f"Error downloading metrics for {model_name}: {e}")
            return None
    
    def upload_config(self, config: Dict[str, Any], run_id: str) -> bool:
        """Upload experiment configuration to S3."""
        try:
            s3_key = f"configs/{run_id}/params.yaml"
            
            # Add metadata
            config['cloud_metadata'] = {
                'uploaded_at': datetime.now().isoformat(),
                'run_id': run_id,
                'bucket': self.bucket_name
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=yaml.dump(config, default_flow_style=False),
                ContentType='application/x-yaml'
            )
            
            print(f"Uploaded config to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading config: {e}")
            return False
    
    def download_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Download experiment configuration from S3."""
        try:
            s3_key = f"configs/{run_id}/params.yaml"
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            config = yaml.safe_load(response['Body'].read())
            
            return config
        except Exception as e:
            print(f"Error downloading config: {e}")
            return None
    
    def upload_training_data(self, data_dir: str, run_id: str) -> bool:
        """Upload processed training data to S3."""
        try:
            for file in os.listdir(data_dir):
                if file.endswith('.csv') or file.endswith('.pkl'):
                    local_file = os.path.join(data_dir, file)
                    s3_key = f"data/{run_id}/processed/{file}"
                    
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            
            print(f"Uploaded training data to s3://{self.bucket_name}/data/{run_id}/processed/")
            return True
        except Exception as e:
            print(f"Error uploading training data: {e}")
            return False
    
    def download_training_data(self, run_id: str, local_dir: str) -> bool:
        """Download processed training data from S3."""
        try:
            prefix = f"data/{run_id}/processed/"
            
            os.makedirs(local_dir, exist_ok=True)
            
            for obj in self.bucket.objects.filter(Prefix=prefix):
                filename = obj.key.split('/')[-1]
                local_file = os.path.join(local_dir, filename)
                
                self.bucket.download_file(obj.key, local_file)
            
            print(f"Downloaded training data to {local_dir}")
            return True
        except Exception as e:
            print(f"Error downloading training data: {e}")
            return False
    
    def list_runs(self, prefix: str = "models/") -> List[str]:
        """List all available runs."""
        try:
            runs = set()
            
            for obj in self.bucket.objects.filter(Prefix=prefix):
                parts = obj.key.split('/')
                if len(parts) >= 2:
                    runs.add(parts[1])  # run_id is second part
            
            return sorted(list(runs))
        except Exception as e:
            print(f"Error listing runs: {e}")
            return []
    
    def list_models(self, run_id: str) -> List[str]:
        """List all models for a specific run."""
        try:
            models = set()
            prefix = f"models/{run_id}/"
            
            for obj in self.bucket.objects.filter(Prefix=prefix):
                parts = obj.key.split('/')
                if len(parts) >= 3:
                    models.add(parts[2])  # model_name is third part
            
            return sorted(list(models))
        except Exception as e:
            print(f"Error listing models for run {run_id}: {e}")
            return []
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary information for a run."""
        try:
            summary = {
                'run_id': run_id,
                'models': self.list_models(run_id),
                'metrics': {},
                'config': self.download_config(run_id)
            }
            
            # Get metrics for each model
            for model in summary['models']:
                metrics = self.download_metrics(run_id, model)
                if metrics:
                    summary['metrics'][model] = metrics
            
            return summary
        except Exception as e:
            print(f"Error getting run summary for {run_id}: {e}")
            return {}
    
    def cleanup_old_runs(self, keep_latest: int = 10) -> int:
        """Clean up old runs, keeping only the latest N."""
        try:
            runs = self.list_runs()
            
            if len(runs) <= keep_latest:
                print(f"Only {len(runs)} runs found, no cleanup needed")
                return 0
            
            # Sort runs by date (assuming run_id contains timestamp)
            runs_to_delete = runs[:-keep_latest]
            deleted_count = 0
            
            for run_id in runs_to_delete:
                # Delete all objects for this run
                prefix = f"models/{run_id}/"
                objects_to_delete = []
                
                for obj in self.bucket.objects.filter(Prefix=prefix):
                    objects_to_delete.append({'Key': obj.key})
                
                if objects_to_delete:
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
                    deleted_count += 1
                    print(f"Deleted run {run_id}")
            
            return deleted_count
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 0

class DVCCloudIntegration:
    """Integrates DVC with cloud storage for remote data versioning."""
    
    def __init__(self, storage_manager: CloudStorageManager):
        self.storage = storage_manager
    
    def setup_dvc_remote(self, remote_name: str = "s3-storage") -> bool:
        """Setup DVC remote storage configuration."""
        try:
            import subprocess
            
            # Add S3 remote
            subprocess.run([
                'dvc', 'remote', 'add', '-d', remote_name, 
                f's3://{self.storage.bucket_name}/dvc-cache'
            ], check=True)
            
            # Configure S3 region
            subprocess.run([
                'dvc', 'remote', 'modify', remote_name, 'region', 
                self.storage.region
            ], check=True)
            
            print(f"DVC remote '{remote_name}' configured for S3")
            return True
        except Exception as e:
            print(f"Error setting up DVC remote: {e}")
            return False
    
    def push_dvc_data(self) -> bool:
        """Push DVC-tracked data to remote storage."""
        try:
            import subprocess
            subprocess.run(['dvc', 'push'], check=True)
            print("DVC data pushed to remote storage")
            return True
        except Exception as e:
            print(f"Error pushing DVC data: {e}")
            return False
    
    def pull_dvc_data(self) -> bool:
        """Pull DVC-tracked data from remote storage."""
        try:
            import subprocess
            subprocess.run(['dvc', 'pull'], check=True)
            print("DVC data pulled from remote storage")
            return True
        except Exception as e:
            print(f"Error pulling DVC data: {e}")
            return False

# Utility functions for cloud integration
def get_cloud_storage_manager() -> CloudStorageManager:
    """Get cloud storage manager from environment variables."""
    bucket_name = os.getenv('S3_BUCKET')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not bucket_name:
        raise ValueError("S3_BUCKET environment variable not set")
    
    return CloudStorageManager(bucket_name, region)

def sync_local_to_cloud(local_dir: str, run_id: str, 
                       storage_manager: CloudStorageManager) -> bool:
    """Sync local training results to cloud."""
    success = True
    
    # Upload models
    models_dir = os.path.join(local_dir, "models")
    if os.path.exists(models_dir):
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                success &= storage_manager.upload_model(model_path, run_id, model_name)
    
    # Upload metrics
    metrics_dir = os.path.join(local_dir, "metrics")
    if os.path.exists(metrics_dir):
        for file in os.listdir(metrics_dir):
            if file.endswith('_metrics.json'):
                with open(os.path.join(metrics_dir, file), 'r') as f:
                    metrics = json.load(f)
                model_name = file.replace('_metrics.json', '')
                success &= storage_manager.upload_metrics(metrics, run_id, model_name)
    
    # Upload processed data
    data_dir = os.path.join(local_dir, "data", "processed")
    if os.path.exists(data_dir):
        success &= storage_manager.upload_training_data(data_dir, run_id)
    
    return success

def sync_cloud_to_local(run_id: str, local_dir: str,
                       storage_manager: CloudStorageManager) -> bool:
    """Sync cloud training results to local."""
    success = True
    
    # Download models
    models = storage_manager.list_models(run_id)
    for model_name in models:
        model_dir = os.path.join(local_dir, "models", model_name)
        success &= storage_manager.download_model(run_id, model_name, model_dir)
    
    # Download training data
    data_dir = os.path.join(local_dir, "data", "processed")
    success &= storage_manager.download_training_data(run_id, data_dir)
    
    return success
