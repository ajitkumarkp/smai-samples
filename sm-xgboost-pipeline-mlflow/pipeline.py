"""
SageMaker Pipeline with Conditional Model Registration (Inline Step Wrapping)

Defines and executes an end-to-end ML pipeline for the Abalone dataset using
SageMaker Pipelines with MLflow experiment tracking.

Pipeline Flow:
    Preprocess → Train → Evaluate → Quality Check → Register (conditional)

Key Features:
    - Uses `step(fn, name=...)` at call time to convert plain functions into
      managed SageMaker Pipeline steps (each runs as a remote training job).
    - Quality gate: model is only registered when MSE and R² meet thresholds.
    - Thresholds exposed as pipeline parameters, overridable per execution.
    - All steps log to nested MLflow runs for unified experiment tracking.

Usage:
    python pipeline.py \\
        --mlflow_tracking_uri <mlflow-arn> \\
        --mlflow_experiment_name <experiment> \\
        --sagemaker_pipeline_name <name> \\
        --mse_threshold 6.0 \\
        --r2_threshold 0.5
"""

import os
import argparse

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.workflow import Pipeline
from sagemaker.mlops.workflow.function_step import step
from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat

from steps.preprocess import preprocess
from steps.train import train
from steps.evaluation import evaluate
from steps.check_quality import check_quality
from steps.register import register

import mlflow

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_tracking_uri', help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment_name', help='MLflow experiment name')
    parser.add_argument('--sagemaker_pipeline_name', help='Name of the SageMaker Pipeline',
                        default="abalone-pipeline-with-quality-gate")
    parser.add_argument('--mse_threshold', type=float, default=6.0,
                        help='Maximum acceptable MSE for model registration (default: 6.0)')
    parser.add_argument('--r2_threshold', type=float, default=0.5,
                        help='Minimum acceptable R² for model registration (default: 0.5)')
    args = parser.parse_args()

    sagemaker_session = Session()

    bucket = sagemaker_session.default_bucket()
    input_path = (f"s3://sagemaker-example-files-prod-{sagemaker_session.boto_region_name}/datasets"
                  f"/tabular/uci_abalone/abalone.csv")

    model_pkg_group_name = "abalone-model-group"

    # Pipeline Parameters
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    mse_threshold_param = ParameterFloat(
        name="MseThreshold",
        default_value=args.mse_threshold
    )
    r2_threshold_param = ParameterFloat(
        name="R2Threshold",
        default_value=args.r2_threshold
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.sagemaker_pipeline_name) as run:
        run_id = run.info.run_id
        print(f"\n{'='*70}")
        print(f"PIPELINE WITH CONDITIONAL MODEL REGISTRATION")
        print(f"{'='*70}")
        print(f"MLflow Run ID: {run_id}")
        print(f"Pipeline Name: {args.sagemaker_pipeline_name}")
        print(f"\nQuality Gate Thresholds:")
        print(f"  • MSE must be < {args.mse_threshold}")
        print(f"  • R²  must be > {args.r2_threshold}")
        print(f"{'='*70}\n")

        # ========================================
        # Pipeline Steps
        # ========================================

        print("Building pipeline steps...")

        # Step 1: Data Preprocessing
        print("  1. Data Preprocessing")
        data = step(preprocess, name="Preprocess_Data")(
            input_path,
            run_id=run_id
        )

        # Step 2: Model Training
        print("  2. Model Training")
        model = step(train, name="Train_Model")(
            train_df=data[0],
            validation_df=data[1],
            run_id=run_id
        )

        # Step 3: Model Evaluation
        print("  3. Model Evaluation")
        evaluation_result = step(evaluate, name="Evaluate_Model")(
            model=model,
            test_df=data[2],
            run_id=run_id
        )

        # Step 4: Quality Check (Conditional Gate)
        print("  4. Quality Check (Quality Gate)")
        quality_check_result = step(check_quality, name="Check_Quality")(
            evaluation=evaluation_result,
            mse_threshold=mse_threshold_param,
            r2_threshold=r2_threshold_param,
            run_id=run_id
        )

        # Step 5: Conditional Model Registration
        # Note: In the new SageMaker SDK, conditional steps work differently
        # We'll implement the logic in the registration step itself
        print("  5. Model Registration (Conditional)")

        # For now, we include all steps in the pipeline
        # The quality check will log whether the model should be registered
        # In a production pipeline, you would use the quality_check_result to
        # conditionally execute the registration step

        model_register = step(register, name="Register_Model")(
            model=model,
            evaluation=evaluation_result,
            quality_check=quality_check_result,  
            model_approval_status=model_approval_status_param,
            model_package_group_name=model_pkg_group_name,
            bucket=bucket,
            run_id=run_id
        )

        # ========================================
        # Create Pipeline
        # ========================================

        print("\nCreating pipeline...")

        pipeline = Pipeline(
            name=args.sagemaker_pipeline_name,
            parameters=[
                model_approval_status_param,
                mse_threshold_param,
                r2_threshold_param
            ],
            steps=[
                # Include all steps - dependencies are automatically tracked
                # The quality_check step will run after evaluation
                # Registration will run after quality check
                model_register
            ],
        )

        print(f"\n{'='*70}")
        print("PIPELINE CONFIGURATION")
        print(f"{'='*70}")
        print(f"Pipeline Name: {args.sagemaker_pipeline_name}")
        print(f"\nParameters:")
        print(f"  • ModelApprovalStatus = {model_approval_status_param.default_value}")
        print(f"  • MseThreshold = {mse_threshold_param.default_value}")
        print(f"  • R2Threshold = {r2_threshold_param.default_value}")
        print(f"\nExecution Flow:")
        print(f"  Preprocess → Train → Evaluate → Quality Check → [Conditional] → Register")
        print(f"\nModel Registration Criteria:")
        print(f"  ✓ Model will be registered if:")
        print(f"      MSE < {mse_threshold_param.default_value} AND R² > {r2_threshold_param.default_value}")
        print(f"  ✗ Model registration will be skipped if criteria not met")
        print(f"{'='*70}\n")

        # Upsert (create or update) the pipeline
        print("Upserting pipeline to SageMaker...")
        pipeline.upsert(
            role_arn=get_execution_role()
        )
        print("✓ Pipeline created/updated successfully!")

        # Start pipeline execution
        print("\nStarting pipeline execution...")
        execution = pipeline.start()
        print("✓ Pipeline execution started!")

        print(f"\n{'='*70}")
        print("EXECUTION STARTED")
        print(f"{'='*70}")
        print(f"Monitor the execution in:")
        print(f"  • SageMaker Studio Console")
        print(f"  • MLflow UI: {args.mlflow_tracking_uri}")
        print(f"\nTo check status via CLI:")
        print(f"  aws sagemaker list-pipeline-executions \\")
        print(f"    --pipeline-name {args.sagemaker_pipeline_name}")
        print(f"{'='*70}\n")
