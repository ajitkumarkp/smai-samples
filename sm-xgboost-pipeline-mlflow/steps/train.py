"""
Model Training Step for the Abalone Pipeline.

Trains an XGBoost gradient-boosted tree regressor to predict abalone age (rings)
from preprocessed features. Uses early stopping on a validation set to prevent
overfitting. All hyperparameters and training curves are logged to MLflow.

Input:
    - train_df: DataFrame with label in column 0, features in remaining columns
    - validation_df: Same structure, used for early stopping evaluation

Output:
    - Trained xgboost.Booster object

MLflow artifacts logged:
    - Full hyperparameter set
    - Training and validation RMSE curves (via autolog)
    - Final metrics and improvement percentage
    - Serialized XGBoost model artifact
"""

import pandas as pd
import os

import xgboost
import mlflow


def train(
    train_df,
    validation_df,
    *,
    num_round=50,
    objective="reg:linear",
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    use_gpu=False,
    experiment_name="abalone-sm-pipeline-exp",
    run_id=None
):
    """
    Train an XGBoost regression model with early stopping.

    Args:
        train_df: Training DataFrame (column 0 = label, rest = features).
        validation_df: Validation DataFrame (same layout).
        num_round: Maximum boosting rounds (default 50).
        objective: XGBoost objective function (default "reg:linear").
        max_depth: Maximum tree depth.
        eta: Learning rate (shrinkage).
        gamma: Minimum loss reduction for a split.
        min_child_weight: Minimum sum of instance weight in a child.
        subsample: Row subsampling ratio per tree.
        use_gpu: If True, uses GPU-accelerated histogram method.
        experiment_name: MLflow experiment name.
        run_id: Parent MLflow run ID for nested logging.

    Returns:
        xgboost.Booster: The trained model.
    """

    # Enable autologging in MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Train", nested=True):
            # enable mlflow autolog
            mlflow.xgboost.autolog()

            y_train = train_df.iloc[:, 0].to_numpy()
            train_df.drop(train_df.columns[0], axis=1, inplace=True)
            x_train = train_df.to_numpy()
            train_dmatrix = xgboost.DMatrix(x_train, label=y_train)

            y_validation = validation_df.iloc[:, 0].to_numpy()
            validation_df.drop(validation_df.columns[0], axis=1, inplace=True)
            x_validation = validation_df.to_numpy()
            validation_dmatrix = xgboost.DMatrix(x_validation, label=y_validation)

            # Log training dataset info
            mlflow.log_param("train_samples", len(x_train))
            mlflow.log_param("validation_samples", len(x_validation))
            mlflow.log_param("num_features", x_train.shape[1])

            param = {
                "objective": objective,
                "max_depth": max_depth,
                "eta": eta,
                "gamma": gamma,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "tree_method": "gpu_hist"
                if use_gpu
                else "hist",  # Use GPU accelerated algorithm
            }

            # Log all hyperparameters explicitly (autolog will also capture these)
            mlflow.log_params({
                "num_boost_round": num_round,
                "objective": objective,
                "max_depth": max_depth,
                "eta": eta,
                "gamma": gamma,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "tree_method": param["tree_method"],
                "use_gpu": use_gpu,
                "early_stopping_rounds": 5
            })

            mlflow.set_tags(
                {
                    'mlflow.source.name': "train.py",
                    'mlflow.source.type': 'TRAIN',
                    'algorithm': 'xgboost',
                }
            )

            evaluation_results = {}  # Store accuracy result
            booster = xgboost.train(
                param,
                train_dmatrix,
                num_round,
                evals=[(train_dmatrix, "train"), (validation_dmatrix, "validation")],
                early_stopping_rounds=5,
                evals_result=evaluation_results,
            )

            # Log final training metrics
            if evaluation_results:
                final_train_rmse = evaluation_results['train']['rmse'][-1]
                final_val_rmse = evaluation_results['validation']['rmse'][-1]
                mlflow.log_metric("final_train_rmse", final_train_rmse)
                mlflow.log_metric("final_validation_rmse", final_val_rmse)

                # Log improvement metrics
                initial_train_rmse = evaluation_results['train']['rmse'][0]
                improvement = ((initial_train_rmse - final_train_rmse) / initial_train_rmse) * 100
                mlflow.log_metric("train_improvement_pct", improvement)

            print(f"✓ Training complete: {booster.num_boosted_rounds()} rounds, "
                  f"final validation RMSE: {final_val_rmse:.4f}")

    return booster
