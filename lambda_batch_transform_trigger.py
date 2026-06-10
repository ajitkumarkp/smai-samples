"""
AWS Lambda: S3-Triggered Batch Transform Using the Latest Approved Model

This Lambda function is designed to be invoked by an S3 event notification
when a new CSV file is uploaded to the `batch-input/` prefix. It:

1. Identifies the latest Approved model from the specified Model Package Group.
2. Creates (or reuses) a SageMaker Model resource pointing to that package.
3. Launches a SageMaker Batch Transform job to generate predictions.
4. Writes inference results to `s3://<bucket>/batch-output/<file>/<timestamp>/`.

Environment Variables (set in Lambda configuration):
    MODEL_PACKAGE_GROUP_NAME  – Model Registry group to query (e.g. "abalone-model-group")
    OUTPUT_BUCKET             – S3 bucket for transform output (defaults to input bucket)
    SAGEMAKER_ROLE            – IAM role ARN passed to the Batch Transform job

Required IAM Permissions (see lambda_execution_policy.json):
    - sagemaker:ListModelPackages, DescribeModel, CreateModel, CreateTransformJob
    - s3:GetObject, ListBucket on the input/output buckets
    - iam:PassRole for the SageMaker execution role
    - CloudWatch Logs write access

Trigger Configuration:
    - S3 Event: s3:ObjectCreated:* on prefix `batch-input/` with suffix `.csv`
"""

import json
import boto3
import os
from datetime import datetime

sagemaker = boto3.client('sagemaker')

def lambda_handler(event, context):
    """
    Lambda function triggered by S3 upload to start SageMaker Batch Transform job.
    Automatically uses the latest approved model from the model package group.
    """
    try:
        # Get S3 event details
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        input_key = s3_event['object']['key']
        
        # Only process CSV files in the 'batch-input/' prefix
        if not input_key.startswith('batch-input/') or not input_key.endswith('.csv'):
            print(f"Skipping non-batch file: {input_key}")
            return {
                'statusCode': 200,
                'body': json.dumps('File skipped - not a batch input CSV')
            }
        
        # Get configuration from environment variables
        model_pkg_group_name = os.environ['MODEL_PACKAGE_GROUP_NAME']
        output_bucket = os.environ.get('OUTPUT_BUCKET', bucket_name)
        sagemaker_role = os.environ['SAGEMAKER_ROLE']
        
        print(f"Looking up latest approved model from: {model_pkg_group_name}")
        
        # DYNAMIC LOOKUP: Get the latest approved model package
        latest_packages = sagemaker.list_model_packages(
            ModelPackageGroupName=model_pkg_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not latest_packages.get('ModelPackageSummaryList'):
            error_msg = f"No approved models found in {model_pkg_group_name}"
            print(f"ERROR: {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        latest_package = latest_packages['ModelPackageSummaryList'][0]
        latest_package_arn = latest_package['ModelPackageArn']
        model_version = latest_package.get('ModelPackageVersion', 'unknown')
        
        print(f"Found latest approved model:")
        print(f"  ARN: {latest_package_arn}")
        print(f"  Version: {model_version}")
        print(f"  Status: {latest_package['ModelApprovalStatus']}")
        
        # Create or reuse model from this package
        # Use a deterministic name based on package ARN to avoid creating duplicates
        package_id = latest_package_arn.split('/')[-1]
        model_name = f"abalone-batch-model-v{model_version}"
        
        try:
            # Check if model already exists
            sagemaker.describe_model(ModelName=model_name)
            print(f"Using existing model: {model_name}")
        except sagemaker.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                # Model doesn't exist, create it
                print(f"Creating new model: {model_name}")
                sagemaker.create_model(
                    ModelName=model_name,
                    Containers=[{
                        'ModelPackageName': latest_package_arn
                    }],
                    ExecutionRoleArn=sagemaker_role
                )
                print(f"Model created: {model_name}")
            else:
                raise
        
        # Generate unique job name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        file_name = input_key.split('/')[-1].replace('.csv', '').replace('_', '-')
        transform_job_name = f"auto-batch-{file_name}-{timestamp}"
        
        # Define input and output S3 paths
        input_s3_uri = f"s3://{bucket_name}/{input_key}"
        output_s3_prefix = f"s3://{output_bucket}/batch-output/{file_name}/{timestamp}/"
        
        print(f"Starting batch transform job: {transform_job_name}")
        print(f"Input: {input_s3_uri}")
        print(f"Output: {output_s3_prefix}")
        
        # Create batch transform job
        response = sagemaker.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            },
            TransformOutput={
                'S3OutputPath': output_s3_prefix,
                'Accept': 'text/csv',
                'AssembleWith': 'Line'
            },
            TransformResources={
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1
            }
        )
        
        print(f"Successfully started transform job: {transform_job_name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Batch transform job started successfully',
                'transformJobName': transform_job_name,
                'modelName': model_name,
                'modelVersion': model_version,
                'modelPackageArn': latest_package_arn,
                'inputFile': input_s3_uri,
                'outputLocation': output_s3_prefix
            })
        }
        
    except Exception as e:
        print(f"Error starting batch transform job: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
