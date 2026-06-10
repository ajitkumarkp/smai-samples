"""
Data Preprocessing Step for the Abalone Pipeline.

This module loads the UCI Abalone dataset from S3, applies feature engineering
(numeric scaling via StandardScaler, categorical one-hot encoding), and produces
a 70/15/15 train/validation/test split. All preprocessing metadata is logged to
MLflow as a nested child run under the pipeline's parent run.

Input:
    - Raw CSV from S3 (headerless, 8 features + 1 label column "rings")

Output:
    - Tuple of three DataFrames: (train_df, validation_df, test_df)
      Each has the label in column 0, followed by transformed features.

MLflow artifacts logged:
    - Input dataset reference
    - Data quality metrics (missing values, duplicates)
    - Per-column statistics (mean, std)
    - Train/validation/test split sizes
"""

import numpy as np
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow

# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merge two dictionaries into a new dict (used for combining dtype specs)."""
    z = x.copy()
    z.update(y)
    return z


def preprocess(raw_data_s3_path: str, experiment_name: str = "abalone-sm-pipeline-exp", run_id: str = None) -> tuple[pd.DataFrame, ...]:
    """
    Load, transform, and split the Abalone dataset.

    Applies a ColumnTransformer pipeline:
      - Numeric features: median imputation → standard scaling
      - Categorical features ("sex"): constant imputation → one-hot encoding

    Args:
        raw_data_s3_path: S3 URI to the raw CSV file (headerless).
        experiment_name: MLflow experiment to log under.
        run_id: Parent MLflow run ID for nested run grouping.

    Returns:
        Tuple of (train_df, validation_df, test_df) as pandas DataFrames.
        Column 0 is the target ("rings"), remaining columns are transformed features.
    """
    df = pd.read_csv(
        raw_data_s3_path,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="DataPreprocessing", nested=True):
            # Enable autologging in MLflow
            mlflow.sklearn.autolog(log_datasets=False)
            dataset = mlflow.data.from_pandas(df, source=raw_data_s3_path)
            mlflow.log_input(dataset, context="feature-engineering")

            # Log input data statistics
            mlflow.log_param("total_samples", len(df))
            mlflow.log_param("feature_count", len(feature_columns_names))
            mlflow.log_param("data_source", raw_data_s3_path)

            # Log data quality metrics
            mlflow.log_metric("missing_values_count", df.isnull().sum().sum())
            mlflow.log_metric("duplicate_rows", df.duplicated().sum())

            # Log basic statistics
            for col in df.select_dtypes(include=[np.number]).columns:
                mlflow.log_metric(f"{col}_mean", df[col].mean())
                mlflow.log_metric(f"{col}_std", df[col].std())

            # Numeric Feature Transformation
            numeric_features = list(feature_columns_names)
            numeric_features.remove("sex")
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            # Categorical Feature Transformation
            categorical_features = ["sex"]
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            preprocess = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            mlflow.set_tags(
                {
                    'mlflow.source.name': "preprocess.py",
                    'mlflow.source.type': 'PREPROCESS',
                }
            )

            y = df.pop("rings")
            X_pre = preprocess.fit_transform(df)
            y_pre = y.to_numpy().reshape(len(y), 1)

            X = np.concatenate((y_pre, X_pre), axis=1)

            np.random.shuffle(X)
            train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

            # Log split sizes
            mlflow.log_param("train_size", len(train))
            mlflow.log_param("validation_size", len(validation))
            mlflow.log_param("test_size", len(test))
            mlflow.log_param("train_ratio", 0.7)
            mlflow.log_param("validation_ratio", 0.15)
            mlflow.log_param("test_ratio", 0.15)

            print(f"✓ Preprocessing complete: {len(train)} train, {len(validation)} validation, {len(test)} test samples")

    return pd.DataFrame(train), pd.DataFrame(validation), pd.DataFrame(test)
