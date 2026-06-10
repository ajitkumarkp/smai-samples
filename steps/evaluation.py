"""
Model Evaluation Step for the Abalone Pipeline.

Runs inference on the held-out test set and computes a comprehensive suite of
regression metrics. Results are structured in a report dict compatible with
SageMaker Model Registry's ModelMetrics format.

Input:
    - Trained xgboost.Booster model
    - test_df: DataFrame with label in column 0

Output:
    - Dictionary with nested regression_metrics (MSE, RMSE, MAE, MAPE, R²)

MLflow artifacts logged:
    - All evaluation metrics (mse, rmse, mae, mape, r2)
    - Prediction distribution statistics
    - Residual distribution statistics
"""

import numpy as np
import os
import xgboost
import mlflow

from sklearn.metrics import mean_squared_error


def evaluate(model, test_df, experiment_name="abalone-sm-pipeline-exp", run_id=None):
    """
    Evaluate the trained model against the test set.

    Computes MSE, RMSE, MAE, MAPE, and R² score. Logs all metrics and
    prediction/residual statistics to MLflow.

    Args:
        model: Trained xgboost.Booster.
        test_df: Test DataFrame (column 0 = label, rest = features).
        experiment_name: MLflow experiment name.
        run_id: Parent MLflow run ID for nested logging.

    Returns:
        dict: Evaluation report with structure:
            {
                "regression_metrics": {
                    "mse": {"value": float, "standard_deviation": float},
                    "rmse": {"value": float},
                    "mae": {"value": float},
                    "mape": {"value": float},
                    "r2_score": {"value": float}
                }
            }
    """

    # Enable autologging in MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Evaluate", nested=True):
            mlflow.autolog()
            y_test = test_df.iloc[:, 0].to_numpy()
            test_df.drop(test_df.columns[0], axis=1, inplace=True)
            x_test = test_df.to_numpy()
            predictions = model.predict(xgboost.DMatrix(x_test))

            # Log test dataset info
            mlflow.log_param("test_samples", len(x_test))

            mse = mean_squared_error(y_test, predictions)
            std = np.std(y_test - predictions)

            # Calculate additional metrics
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - predictions))
            residuals = y_test - predictions

            # Calculate percentage errors
            mape = np.mean(np.abs(residuals / y_test)) * 100  # Mean Absolute Percentage Error

            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            report_dict = {
                "regression_metrics": {
                    "mse": {"value": mse, "standard_deviation": std},
                    "rmse": {"value": rmse},
                    "mae": {"value": mae},
                    "mape": {"value": mape},
                    "r2_score": {"value": r2},
                },
            }

            mlflow.set_tags(
                {
                    'mlflow.source.name': "evaluation.py",
                    'mlflow.source.type': 'EVALUATION',
                }
            )

            # Log all metrics
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mse_std", std)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_mape", mape)
            mlflow.log_metric("test_r2", r2)

            # Log prediction statistics
            mlflow.log_metric("predictions_mean", np.mean(predictions))
            mlflow.log_metric("predictions_std", np.std(predictions))
            mlflow.log_metric("predictions_min", np.min(predictions))
            mlflow.log_metric("predictions_max", np.max(predictions))

            # Log residual statistics
            mlflow.log_metric("residuals_mean", np.mean(residuals))
            mlflow.log_metric("residuals_std", np.std(residuals))

            print(f"✓ Evaluation complete:")
            print(f"  MSE: {mse:.4f} (±{std:.4f})")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")

    return report_dict