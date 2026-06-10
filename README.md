# SageMaker Pipeline with Quality Gate and MLflow Tracking

An end-to-end MLOps workflow that trains an XGBoost regression model on the [UCI Abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone), evaluates it against configurable quality thresholds, and conditionally registers it in the SageMaker Model Registry — all tracked via MLflow.

## Key Learning Objectives

This example demonstrates two important MLOps patterns:

### 1. The `@step` decorator: Pipelines without rewriting your code

The core idea is that the **`@step` decorator lets you turn existing Python functions into SageMaker Pipeline steps with minimal code changes**. You don't need to restructure your ML code around pipeline-specific abstractions — just add `@step` to your functions and they become managed pipeline steps.

To illustrate this, the project includes **two implementations** of the same pipeline:

| Approach | Pipeline file | Step files | Key point |
|----------|--------------|------------|-----------|
| **Traditional (inline)** | `pipeline.py` | `steps/*.py` | Step wrapping happens in the pipeline script with `step(fn, name=...)`. Your functions stay undecorated but the pipeline orchestration code is more verbose. |
| **`@step` decorator (new)** | `pipeline_decorated.py` | `steps/*_decorated.py` | Add `@step(name=...)` to function definitions. Pipeline code becomes clean Python — just call your functions and the SDK figures out the DAG from data dependencies. |

Both produce **functionally identical pipelines**. Compare the two to see how little your existing ML code needs to change to become a production pipeline.

### 2. S3-triggered Batch Transform: Event-driven inference

The project shows how to build a **fully automated inference pipeline** triggered by S3 uploads:

```
CSV uploaded to S3  →  S3 Event Notification  →  Lambda Function  →  SageMaker Batch Transform  →  Predictions in S3
```

Key aspects:
- **Zero manual intervention** — upload a file, get predictions automatically
- **Always uses the latest approved model** — Lambda dynamically queries the Model Registry
- **Full lineage preserved** — model package → model → batch transform job
- **Pay-per-use** — no persistent endpoint; compute spins up only when needed

The implementation is in `lambda_batch_transform_trigger.py` with IAM policies ready to deploy.

---

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│  Preprocess │───▶│    Train    │───▶│  Evaluate   │───▶│ Quality Check │───▶│   Register   │
│   (data)    │    │  (XGBoost)  │    │  (metrics)  │    │  (gate logic) │    │ (conditional)│
└─────────────┘    └─────────────┘    └─────────────┘    └───────────────┘    └──────────────┘
       │                  │                  │                    │                     │
       └──────────────────┴──────────────────┴────────────────────┴─────────────────────┘
                                    MLflow Tracking (nested runs)
```

## Pipeline Steps

| Step | File | Description |
|------|------|-------------|
| 1. Preprocess | `steps/preprocess.py` | Loads Abalone CSV from S3, applies numeric scaling + categorical one-hot encoding, splits into 70/15/15 train/validation/test |
| 2. Train | `steps/train.py` | Trains an XGBoost regressor with early stopping (default 50 rounds, stops after 5 rounds without improvement) |
| 3. Evaluate | `steps/evaluation.py` | Computes MSE, RMSE, MAE, MAPE, and R² on the test set |
| 4. Quality Check | `steps/check_quality.py` | Compares evaluation metrics against thresholds (MSE < 6.0 AND R² > 0.5) and emits a pass/fail decision |
| 5. Register | `steps/register.py` | Packages the model as `model.tar.gz`, uploads to S3, registers in SageMaker Model Registry, and logs the model to MLflow |

## Quality Gate

The pipeline implements a quality gate between evaluation and registration:

- **MSE Threshold** (default 6.0): model must have MSE below this value
- **R² Threshold** (default 0.5): model must have R² above this value

These are exposed as `ParameterFloat` pipeline parameters, so they can be overridden per execution without code changes.

## Project Structure

```
sm-xgboost-pipeline-mlflow/
├── runme.ipynb                      # Interactive notebook: local test + remote pipeline execution
├── pipeline.py                      # Pipeline definition (traditional inline step wrapping)
├── pipeline_decorated.py            # Pipeline definition (@step decorator — recommended)
├── config.yaml                      # SageMaker SDK remote function config
├── requirements.txt                 # Dependencies installed inside remote jobs
├── local-requirements.txt           # Dependencies for local development
├── steps/
│   ├── __init__.py
│   ├── preprocess.py                # Data loading & feature engineering
│   ├── preprocess_decorated.py      # Same logic with @step decorator
│   ├── train.py                     # XGBoost training
│   ├── train_decorated.py
│   ├── evaluation.py                # Model evaluation metrics
│   ├── evaluation_decorated.py
│   ├── check_quality.py             # Quality gate logic
│   ├── check_quality_decorated.py
│   ├── register.py                  # Model packaging & registry
│   └── register_decorated.py
├── lambda_batch_transform_trigger.py  # Lambda: S3-triggered batch inference
├── lambda_execution_policy.json       # IAM policy for the Lambda function
├── lambda_trust_policy.json           # Trust policy for the Lambda execution role
├── abalone.csv                        # Sample dataset (local copy)
├── batch_2.csv                        # Sample batch input for transform testing
└── batch_output.csv.out               # Sample batch output
```

## Prerequisites

- **Python**: 3.10+
- **SageMaker Studio**: JupyterLab space with an execution role
- **MLflow App**: An active MLflow tracking server in SageMaker (auto-detected in the notebook)
- **IAM Role**: SageMaker execution role with access to S3, Model Registry, and MLflow

## Quick Start

### 1. Run interactively (notebook)

Open `runme.ipynb` and run all cells. The notebook will:

1. Install dependencies
2. Detect your MLflow tracking server
3. Generate `config.yaml` with environment-specific settings
4. Execute all pipeline steps locally for fast validation
5. Submit the pipeline to SageMaker for managed execution
6. Deploy the model and run batch inference

### 2. Run via CLI

```bash
python pipeline_decorated.py \
  --mlflow_tracking_uri arn:aws:sagemaker:<region>:<account>:mlflow-app/<app-id> \
  --mlflow_experiment_name abalone-sm-pipeline-exp \
  --sagemaker_pipeline_name abalone-pipeline-with-quality-gate \
  --mse_threshold 6.0 \
  --r2_threshold 0.5
```

## Configuration

### `config.yaml`

Controls the remote function execution environment:

| Key | Purpose |
|-----|---------|
| `InstanceType` | Compute for each step (default `ml.m5.xlarge`) |
| `EnvironmentVariables` | Passes MLflow URI, experiment name, and region into remote jobs |
| `Dependencies` | Points to `requirements.txt` for pip installs inside the container |
| `IncludeLocalWorkDir` | Ships local Python files alongside the remote function |
| `CustomFileFilter` | Excludes data, models, notebooks, and cache from the upload |

### Pipeline Parameters (runtime-configurable)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ModelApprovalStatus` | String | `PendingManualApproval` | Initial approval state for registered models |
| `MseThreshold` | Float | `6.0` | Max acceptable MSE |
| `R2Threshold` | Float | `0.5` | Min acceptable R² |

## S3-Triggered Batch Transform

`lambda_batch_transform_trigger.py` implements the event-driven batch inference pattern:

1. Upload a CSV to `s3://<bucket>/batch-input/`
2. S3 event notification triggers the Lambda function
3. Lambda looks up the latest **Approved** model from the Model Registry
4. Lambda starts a SageMaker Batch Transform job
5. Predictions are written to `s3://<bucket>/batch-output/<filename>/<timestamp>/`

Deploy with the provided IAM policies (`lambda_execution_policy.json`, `lambda_trust_policy.json`).

## MLflow Tracking

All steps log to a shared parent MLflow run using nested child runs:

```
Parent Run (pipeline name)
├── DataPreprocessing  — input stats, split sizes, transformers
├── Train              — hyperparameters, training curves, model artifact
├── Evaluate           — MSE, RMSE, MAE, MAPE, R², residual stats
├── Quality_Check      — thresholds, pass/fail decision
└── Register           — model URI, package ARN, approval status
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sagemaker` | 3.7.1 | Pipeline SDK |
| `sagemaker-mlops` | 1.7.1 | `Pipeline`, `step` function |
| `xgboost` | 3.0.5 | Model training |
| `mlflow` | <4 | Experiment tracking |
| `scikit-learn` | latest | Preprocessing transformers |
| `s3fs` | 0.4.2 | S3 file operations |
