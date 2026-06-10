"""
Model Registration Step for the Abalone Pipeline.

Packages the trained XGBoost model and registers it in the SageMaker Model Registry.
Also logs the model to MLflow for dual lineage tracking.

Workflow:
    1. Upload evaluation report JSON to S3
    2. Log the XGBoost model to MLflow's artifact store
    3. Save the native XGBoost model and create a model.tar.gz archive
    4. Upload model.tar.gz to S3
    5. Register a Model Package in SageMaker Model Registry with:
       - Container image (XGBoost 3.0-5 inference image)
       - Model data URL (S3 path to model.tar.gz)
       - Model metrics reference (evaluation report)
       - Supported instance types for inference and transform

Input:
    - Trained xgboost.Booster model
    - Evaluation report dict
    - Quality check result (optional, for audit logging)
    - Model approval status (pipeline parameter)
    - Model package group name

Output:
    - Model Package ARN (string)

MLflow artifacts logged:
    - Evaluation metrics, model URI, package ARN
    - Quality check decision (if provided)
    - Registration parameters (image, instances, approval status)
"""

import json
import os
import tempfile
import tarfile

import numpy as np
import s3fs as s3fs
from sagemaker.core.model_metrics import ModelMetrics, MetricsSource
from sagemaker.core.s3.utils import s3_path_join
from sagemaker.core.common_utils import unique_name_from_base
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.model_registry import create_model_package_from_containers

import mlflow


def register(
    model,
    evaluation,
    model_approval_status,
    model_package_group_name,
    bucket,
    quality_check=None,
    experiment_name="abalone-sm-pipeline-exp",
    run_id=None
):
    """
    Package and register the model in SageMaker Model Registry and MLflow.

    Args:
        model: Trained xgboost.Booster to register.
        evaluation: Evaluation report dict (from evaluate step).
        model_approval_status: Initial approval state (e.g. "PendingManualApproval").
        model_package_group_name: Model Registry group to register under.
        bucket: S3 bucket for model and report storage.
        quality_check: Optional quality gate result dict (logged for audit trail).
        experiment_name: MLflow experiment name.
        run_id: Parent MLflow run ID for nested logging.

    Returns:
        str: The ARN of the registered SageMaker Model Package.
    """
    sagemaker_session = Session()
    region = sagemaker_session.boto_region_name

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Register", nested=True) as nested_run:
            # Log quality check result if provided
            if quality_check is not None:
                print(f"\n{'='*60}")
                print("QUALITY CHECK RESULT IN REGISTRATION")
                print(f"{'='*60}")
                should_register = quality_check.get('should_register', True)
                decision_reason = quality_check.get('decision_reason', 'N/A')
                print(f"Should Register: {should_register}")
                print(f"Reason: {decision_reason}")
                print(f"{'='*60}\n")

                # Log to MLflow
                mlflow.log_param('quality_check_passed', should_register)
                mlflow.log_param('quality_check_reason', decision_reason)

            # Upload evaluation report to s3
            eval_file_name = unique_name_from_base("evaluation")
            eval_report_s3_uri = s3_path_join(
                "s3://", bucket, f"evaluation-report/{eval_file_name}.json"
            )

            mlflow.log_param('eval_report_s3_uri', eval_report_s3_uri)
            s3_fs = s3fs.S3FileSystem()
            eval_report_str = json.dumps(evaluation)
            with s3_fs.open(eval_report_s3_uri, "wb") as file:
                file.write(eval_report_str.encode("utf-8"))

            # Log evaluation metrics from the report
            if "regression_metrics" in evaluation:
                for metric_name, metric_data in evaluation["regression_metrics"].items():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        mlflow.log_metric(f"eval_{metric_name}", metric_data["value"])

            model_metrics = ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=eval_report_s3_uri,
                    content_type="application/json",
                )
            )

            # 1. Log model to MLflow
            model_info = mlflow.xgboost.log_model(model, artifact_path="model")

            # 2. Save native XGBoost model and create model.tar.gz for SageMaker
            tmp_dir = tempfile.mkdtemp()
            native_model_path = os.path.join(tmp_dir, "xgboost-model")
            model.save_model(native_model_path)

            model_tar_path = tempfile.mktemp(suffix=".tar.gz")
            with tarfile.open(model_tar_path, "w:gz") as tar:
                tar.add(native_model_path, arcname="xgboost-model")

            model_s3_uri = s3_path_join("s3://", bucket, f"models/{unique_name_from_base('model')}/model.tar.gz")
            with s3_fs.open(model_s3_uri, "wb") as f:
                with open(model_tar_path, "rb") as local_f:
                    f.write(local_f.read())

            # Log model artifacts
            mlflow.log_param('model_s3_uri', model_s3_uri)
            mlflow.log_param('model_format', 'xgboost')

            # 3. Register model package directly via core API (no sagemaker.serve dependency)
            image_uri = image_uris.retrieve(
                framework="xgboost",
                region=region,
                version="3.0-5",
            )
            container_def = {
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri,
            }

            # Log registration parameters
            mlflow.log_params({
                'model_package_group_name': model_package_group_name,
                'approval_status': model_approval_status,
                'inference_instances': 'ml.t2.medium,ml.m5.xlarge',
                'transform_instances': 'ml.m5.xlarge',
                'container_image': image_uri,
                'xgboost_version': '3.0-5',
            })

            response = create_model_package_from_containers(
                sagemaker_session=sagemaker_session,
                containers=[container_def],
                content_types=["text/csv"],
                response_types=["text/csv"],
                inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"],
                model_package_group_name=model_package_group_name,
                approval_status=model_approval_status,
                model_metrics=model_metrics._to_request_dict(),
            )
            model_package_arn = response.get("ModelPackageArn")

            mlflow.set_tags({
                'mlflow.source.name': "register.py",
                'mlflow.source.type': 'REGISTER',
                'model.framework': 'xgboost',
                'model.type': 'regression',
            })

            mlflow.log_param('mlflow_model_uri', model_info.model_uri)
            mlflow.log_param('model_package_arn', model_package_arn)

            # Log artifact URLs
            mlflow.log_text(model_package_arn, "model_package_arn.txt")

            print(f"✓ Model registered successfully:")
            print(f"  Model Package ARN: {model_package_arn}")
            print(f"  Approval Status: {model_approval_status}")
            print(f"  MLflow URI: {model_info.model_uri}")

    return model_package_arn
