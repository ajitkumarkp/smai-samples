"""
Pipeline step functions for the Abalone MLOps workflow.

Each module in this package implements one logical step of the pipeline:
    - preprocess: Data loading, feature engineering, and train/val/test splitting
    - train: XGBoost model training with early stopping
    - evaluation: Regression metric computation on the test set
    - check_quality: Quality gate comparing metrics against thresholds
    - register: Model packaging and SageMaker Model Registry registration

Two variants of each step exist:
    - Plain functions (e.g. preprocess.py) — wrapped with step() at call time
    - Decorated functions (e.g. preprocess_decorated.py) — @step applied at definition

All steps log to MLflow as nested child runs under a shared parent run.
"""
