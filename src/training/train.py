import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import os
import logging
import joblib  # For saving models
from time import time

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Data folder paths
script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"

# Constants
INPUT_FEATURES = [
    "initial_batter",
    "initial_bowler",
    "num_batsmen",
    "num_bowlers",
    "num_deliveries",
    "remaining_wickets",
    "remaining_overs",
]
TARGET = "runs"
GROUP_COL = "matchid"


def validate_training_data(df: pd.DataFrame):
    """
    Validate the structure and contents of the training data.
    """
    required_columns = INPUT_FEATURES + [TARGET, GROUP_COL]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Training data is empty.")

    if df[required_columns].isnull().sum().sum() > 0:
        raise ValueError("Training data contains missing values.")


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate the model on the given dataset and log the metrics.
    """
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    logging.info(f"{dataset_name} MAE: {mae}")
    logging.info(f"{dataset_name} RMSE: {rmse}")
    return mae, rmse


def main():
    train_file = os.path.join(data_folder, "training", "training_data.parquet")

    # Load data
    logging.info(f"Loading data from {train_file}")
    df = pd.read_parquet(train_file)

    # Validate data
    logging.info("Validating training data")
    validate_training_data(df)

    # Features and target
    X = df[INPUT_FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COL]

    # Grouped split by matchid
    logging.info("Splitting data into train and test sets")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model
    logging.info("Starting model training")
    start_time = time()
    model = RandomForestRegressor(n_estimators=10, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    end_time = time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

    # Evaluate model
    logging.info("Evaluating model on training set")
    evaluate_model(model, X_train, y_train, dataset_name="Training")

    logging.info("Evaluating model on test set")
    evaluate_model(model, X_test, y_test, dataset_name="Test")

    # Save the model
    model_file = os.path.join(
        script_folder.parent, "model_package", "expected_runs_model.pkl"
    )
    logging.info(f"Saving model to {model_file}")
    joblib.dump(model, model_file)


if __name__ == "__main__":
    main()
