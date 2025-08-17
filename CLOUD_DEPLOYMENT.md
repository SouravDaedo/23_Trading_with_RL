# Cloud Deployment Guide for Trading System

This guide covers the best practices for deploying your trading system to the cloud for 24/7 operation.

## 🏆 Recommended Architecture: AWS MWAA + ECS

### **Why This Setup?**
- **Managed Airflow (MWAA)**: No server maintenance, auto-scaling, 99.9% uptime
- **ECS Fargate**: Serverless containers, pay-per-use, auto-scaling
- **S3 Storage**: Unlimited, versioned storage for models and data
- **Cost**: ~$200-500/month depending on usage

## 🚀 Quick Deployment Steps

### **1. Prerequisites**
```bash
# Install required tools
pip install boto3 terraform awscli

# Configure AWS credentials
aws configure
```

### **2. Deploy Infrastructure**
```bash
cd deploy/aws/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Deploy infrastructure
terraform apply
```

### **3. Build and Push Docker Image**
```bash
# Build training container
docker build -f deploy/aws/docker/Dockerfile -t trading-system .

# Tag for ECR
docker tag trading-system:latest <ECR_URL>:latest

# Push to ECR
docker push <ECR_URL>:latest
```

### **4. Deploy Airflow DAGs**
```bash
# Upload DAGs to S3
aws s3 cp deploy/aws/airflow_cloud_dags/ s3://<AIRFLOW_BUCKET>/dags/ --recursive

# Upload requirements
aws s3 cp requirements.txt s3://<AIRFLOW_BUCKET>/requirements.txt
```

## Architecture Components

### **AWS MWAA (Managed Airflow)**
- **Purpose**: Orchestration and scheduling
- **Features**: Auto-scaling, monitoring, logging
- **Cost**: ~$1/hour for small environment

### **ECS Fargate (Training Workloads)**
- **Purpose**: Run training containers
- **Features**: Serverless, auto-scaling, GPU support
- **Cost**: Pay per vCPU/memory hour

### **S3 (Storage)**
- **Purpose**: Models, data, metrics, configs
- **Features**: Versioning, lifecycle policies, encryption
- **Cost**: ~$0.023/GB/month

### **ECR (Container Registry)**
- **Purpose**: Store Docker images
- **Features**: Vulnerability scanning, lifecycle policies
- **Cost**: ~$0.10/GB/month

##  Configuration

### **Set Airflow Variables**
```bash
# Required variables for cloud deployment
aws mwaa put-environment --name trading-system-airflow --airflow-configuration-options '{
  "AWS_REGION": "us-east-1",
  "ECS_CLUSTER": "trading-system-cluster",
  "TASK_DEFINITION": "trading-system-training",
  "S3_BUCKET": "trading-system-data-bucket",
  "SUBNET_IDS": "subnet-123,subnet-456",
  "SECURITY_GROUP_IDS": "sg-123"
}'
```

### **Environment Variables in ECS**
```json
{
  "environment": [
    {"name": "S3_BUCKET", "value": "trading-system-data"},
    {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"},
    {"name": "RUN_ID", "value": "{{ run_id }}"},
    {"name": "PYTHONPATH", "value": "/app"}
  ]
}
```

##  Cost Breakdown

### **Monthly Costs (Estimated)**

| Component | Small Setup | Medium Setup | Large Setup |
|-----------|-------------|--------------|-------------|
| **MWAA** | $150 | $300 | $600 |
| **ECS Training** | $50 | $200 | $800 |
| **S3 Storage** | $10 | $50 | $200 |
| **ECR** | $5 | $20 | $50 |
| **Data Transfer** | $10 | $30 | $100 |
| **Total** | **$225** | **$600** | **$1,750** |

### **Cost Optimization Tips**
- Use Spot instances for training (50-70% savings)
- Implement S3 lifecycle policies
- Schedule training during off-peak hours
- Use reserved capacity for predictable workloads

## 🔄 Deployment Workflow

### **Development to Production**
```
Local Development → Docker Build → ECR Push → ECS Deploy → MWAA Schedule
```

### **CI/CD Pipeline**
```yaml
# GitHub Actions example
name: Deploy Trading System
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Configure AWS
      uses: aws-actions/configure-aws-credentials@v1
    - name: Build and push Docker image
      run: |
        docker build -t trading-system .
        docker tag trading-system:latest $ECR_URI:latest
        docker push $ECR_URI:latest
    - name: Update ECS service
      run: aws ecs update-service --cluster trading-system --service training-service --force-new-deployment
```

## 📈 Monitoring and Alerting

### **CloudWatch Integration**
- **Metrics**: Task success/failure rates, execution times
- **Logs**: Centralized logging from all containers
- **Alarms**: Automated alerts for failures

### **Custom Metrics**
```python
# In your training code
import boto3

cloudwatch = boto3.client('cloudwatch')

# Send custom metrics
cloudwatch.put_metric_data(
    Namespace='TradingSystem',
    MetricData=[
        {
            'MetricName': 'PortfolioReturn',
            'Value': portfolio_return,
            'Unit': 'Percent'
        }
    ]
)
```

## 🔒 Security Best Practices

### **IAM Roles and Policies**
- **Principle of least privilege**: Only necessary permissions
- **Service-specific roles**: Separate roles for MWAA, ECS, Lambda
- **Cross-account access**: For multi-environment setups

### **Network Security**
- **VPC**: Private subnets for compute resources
- **Security Groups**: Restrictive ingress/egress rules
- **NAT Gateway**: Outbound internet access for private resources

### **Data Encryption**
- **S3**: Server-side encryption (SSE-S3 or SSE-KMS)
- **ECS**: Encryption at rest and in transit
- **Secrets**: AWS Secrets Manager for API keys

## 🚨 Disaster Recovery

### **Backup Strategy**
- **S3 Cross-Region Replication**: Automatic backup to secondary region
- **Database Backups**: RDS automated backups (if using database)
- **Infrastructure as Code**: Terraform state in S3 with versioning

### **Recovery Procedures**
```bash
# Restore from backup region
aws s3 sync s3://backup-bucket/models/ s3://primary-bucket/models/

# Redeploy infrastructure
terraform apply -var="region=us-west-2"

# Update DNS/load balancer
aws route53 change-resource-record-sets --hosted-zone-id Z123 --change-batch file://failover.json
```

## 🎯 Alternative Cloud Options

### **Google Cloud Platform**
- **Cloud Composer**: Managed Airflow
- **Cloud Run**: Serverless containers
- **Cloud Storage**: Object storage
- **Cost**: Similar to AWS, better for ML workloads

### **Microsoft Azure**
- **Data Factory**: Workflow orchestration
- **Container Instances**: Serverless containers
- **Blob Storage**: Object storage
- **Cost**: Competitive pricing, good Windows integration

### **Multi-Cloud Strategy**
```python
# Cloud-agnostic storage interface
class CloudStorage:
    def __init__(self, provider='aws'):
        if provider == 'aws':
            self.client = boto3.client('s3')
        elif provider == 'gcp':
            self.client = storage.Client()
        elif provider == 'azure':
            self.client = BlobServiceClient()
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **ECS Task Failures**
   ```bash
   # Check task logs
   aws logs get-log-events --log-group-name /ecs/trading-system
   
   # Check task definition
   aws ecs describe-task-definition --task-definition trading-system-training
   ```

2. **MWAA Connection Issues**
   ```bash
   # Check environment status
   aws mwaa get-environment --name trading-system-airflow
   
   # View Airflow logs
   aws logs get-log-events --log-group-name airflow-trading-system-airflow-Task
   ```

3. **S3 Permission Errors**
   ```bash
   # Test S3 access
   aws s3 ls s3://trading-system-data/
   
   # Check IAM policies
   aws iam get-role-policy --role-name trading-ecs-task-role --policy-name S3Access
   ```

### **Performance Optimization**

1. **Container Optimization**
   ```dockerfile
   # Multi-stage build for smaller images
   FROM python:3.9-slim as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   FROM python:3.9-slim
   COPY --from=builder /root/.local /root/.local
   ```

2. **Resource Allocation**
   ```json
   {
     "cpu": "2048",
     "memory": "8192",
     "ephemeralStorage": {"sizeInGiB": 50}
   }
   ```

3. **Parallel Training**
   ```python
   # Use multiple ECS tasks for parallel training
   for symbol in symbols:
       ecs_task = EcsRunTaskOperator(
           task_id=f'train_{symbol}',
           overrides={'containerOverrides': [{'environment': [
               {'name': 'SYMBOL', 'value': symbol}
           ]}]}
       )
   ```

## 📋 Deployment Checklist

### **Pre-Deployment**
- [ ] AWS credentials configured
- [ ] Terraform installed and initialized
- [ ] Docker image built and tested locally
- [ ] S3 buckets created with proper permissions
- [ ] VPC and networking configured

### **Deployment**
- [ ] Infrastructure deployed via Terraform
- [ ] Docker image pushed to ECR
- [ ] ECS task definition updated
- [ ] Airflow DAGs uploaded to S3
- [ ] MWAA environment configured

### **Post-Deployment**
- [ ] Test DAG execution in Airflow UI
- [ ] Verify ECS tasks can start successfully
- [ ] Check S3 permissions and data access
- [ ] Set up CloudWatch alarms
- [ ] Configure backup and monitoring

### **Production Readiness**
- [ ] Load testing completed
- [ ] Disaster recovery plan tested
- [ ] Security review completed
- [ ] Cost monitoring configured
- [ ] Documentation updated

This cloud deployment provides enterprise-grade reliability, scalability, and cost-effectiveness for your trading system while maintaining the flexibility to scale up or down based on your needs.
