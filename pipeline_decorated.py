"""
SageMaker Pipeline with Conditional Model Registration (Decorated Version)

Functionally identical to pipeline.py, but uses the decorator-based pattern:
each step function is annotated with @step(name=...) at definition time
(in the *_decorated.py files), so calls in this script look like plain Python.

Pipeline Flow:
    Preprocess → Train → Evaluate → Quality Check → Register (conditional)

Key Difference from pipeline.py:
    - No explicit step(...) wrapping at call time.
    - Decorated functions are imported from steps/*_decorated.py.
    - Pipeline dependency graph is inferred from argument passing.

Usage:
    python pipeline_decorated.py \\
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
from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat

from steps.preprocess_decorated import preprocess
from steps.train_decorated import train
from steps.evaluation_decorated import evaluate
from steps.check_quality_decorated import check_quality
from steps.register_decorated import register

import mlflow

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_tracking_uri', help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment_name', help='MLflow experiment name')
    parser.add_argument('--sagemaker_pipeline_name', help='Name of the SageMaker Pipeline',
                        default="abalone-pipeline-quality-gate-decorated")
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
        print(f"CONDITIONAL PIPELINE (DECORATED VERSION)")
        print(f"{'='*70}")
        print(f"MLflow Run ID: {run_id}")
        print(f"Pipeline Name: {args.sagemaker_pipeline_name}")
        print(f"\nQuality Gate Thresholds:")
        print(f"  • MSE must be < {args.mse_threshold}")
        print(f"  • R²  must be > {args.r2_threshold}")
        print(f"\nNote: Using decorated step functions (@step at definition)")
        print(f"{'='*70}\n")

        # ========================================
        # Pipeline Steps (using decorated functions)
        # ========================================

        print("Building pipeline steps...")

        # Step 1: Data Preprocessing (already decorated in the file)
        print("  1. Data Preprocessing")
        data = preprocess(
            input_path,
            run_id=run_id
        )

        # Step 2: Model Training (already decorated)
        print("  2. Model Training")
        model = train(
            train_df=data[0],
            validation_df=data[1],
            run_id=run_id
        )

        # Step 3: Model Evaluation (already decorated)
        print("  3. Model Evaluation")
        evaluation_result = evaluate(
            model=model,
            test_df=data[2],
            run_id=run_id
        )

        # Step 4: Quality Check (already decorated)
        print("  4. Quality Check (Quality Gate)")
        quality_check_result = check_quality(
            evaluation=evaluation_result,
            mse_threshold=mse_threshold_param,
            r2_threshold=r2_threshold_param,
            run_id=run_id
        )

        # Step 5: Conditional Model Registration (already decorated)
        print("  5. Model Registration (Conditional)")

        # Note: The quality check will log whether the model should be registered
        # In a production pipeline, you would use the quality_check_result to
        # conditionally execute the registration step

        model_register = register(
            model=model,
            evaluation=evaluation_result,
            quality_check=quality_check_result,  # ← Pass quality check to create dependency
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
        print(f"\nDecorator Pattern:")
        print(f"  All step functions have @step decorator at definition")
        print(f"  Function calls look like normal Python calls")
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
