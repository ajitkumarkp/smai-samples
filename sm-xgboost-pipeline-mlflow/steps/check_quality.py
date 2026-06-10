"""
Model Quality Check Step for the Abalone Pipeline.

Implements a configurable quality gate between evaluation and model registration.
Extracts MSE and R² from the evaluation report and compares them against
user-defined thresholds. The decision (pass/fail) is logged to MLflow and
returned as a structured dict consumed by the downstream registration step.

Quality Criteria (both must pass):
    - MSE < mse_threshold (default 6.0)
    - R²  > r2_threshold (default 0.5)

Input:
    - evaluation: Dict returned by the evaluate() step
    - mse_threshold: Maximum acceptable MSE (pipeline parameter)
    - r2_threshold: Minimum acceptable R² (pipeline parameter)

Output:
    - Dict with should_register flag, actual metrics, and decision reason string

MLflow artifacts logged:
    - Threshold parameters
    - Actual model metrics
    - Pass/fail flags for each check
    - JSON artifact with full decision payload
"""

import json
import os
import mlflow


def check_quality(
    evaluation,
    mse_threshold=6.0,
    r2_threshold=0.5,
    experiment_name="abalone-sm-pipeline-exp",
    run_id=None
):
    """
    Check if model quality meets the registration criteria.

    Args:
        evaluation: Evaluation results dict with regression_metrics
        mse_threshold: Maximum acceptable MSE (default: 6.0)
        r2_threshold: Minimum acceptable R² score (default: 0.5)
        experiment_name: MLflow experiment name
        run_id: Parent MLflow run ID

    Returns:
        dict: {
            'should_register': bool,
            'mse': float,
            'r2': float,
            'decision_reason': str
        }
    """
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', ''))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Quality_Check", nested=True):
            mlflow.set_tags({
                'mlflow.source.name': "check_quality.py",
                'mlflow.source.type': 'QUALITY_CHECK',
            })

            # Extract metrics from evaluation
            regression_metrics = evaluation.get('regression_metrics', {})
            mse = regression_metrics.get('mse', {}).get('value', float('inf'))
            r2 = regression_metrics.get('r2_score', {}).get('value', 0.0)

            # Log thresholds
            mlflow.log_param('mse_threshold', mse_threshold)
            mlflow.log_param('r2_threshold', r2_threshold)

            # Log actual metrics
            mlflow.log_metric('model_mse', mse)
            mlflow.log_metric('model_r2', r2)

            # Decision logic
            mse_passed = mse < mse_threshold
            r2_passed = r2 > r2_threshold
            should_register = mse_passed and r2_passed

            # Log decisions
            mlflow.log_metric('mse_check_passed', 1 if mse_passed else 0)
            mlflow.log_metric('r2_check_passed', 1 if r2_passed else 0)
            mlflow.log_metric('quality_check_passed', 1 if should_register else 0)

            # Create decision reason
            reasons = []
            if mse_passed:
                reasons.append(f"✓ MSE {mse:.3f} < {mse_threshold}")
            else:
                reasons.append(f"✗ MSE {mse:.3f} >= {mse_threshold}")

            if r2_passed:
                reasons.append(f"✓ R² {r2:.3f} > {r2_threshold}")
            else:
                reasons.append(f"✗ R² {r2:.3f} <= {r2_threshold}")

            decision_reason = " | ".join(reasons)

            result = {
                'should_register': should_register,
                'mse': mse,
                'r2': r2,
                'mse_passed': mse_passed,
                'r2_passed': r2_passed,
                'decision_reason': decision_reason
            }

            # Log the decision
            mlflow.log_dict(result, 'quality_check_result.json')

            print(f"\n{'='*60}")
            print("MODEL QUALITY CHECK")
            print(f"{'='*60}")
            print(f"Thresholds:")
            print(f"  MSE Threshold: {mse_threshold}")
            print(f"  R² Threshold:  {r2_threshold}")
            print(f"\nModel Performance:")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²:  {r2:.4f}")
            print(f"\nDecision:")
            for reason in reasons:
                print(f"  {reason}")
            print(f"\nResult: {'✓ REGISTER MODEL' if should_register else '✗ SKIP REGISTRATION'}")
            print(f"{'='*60}\n")

    return result
