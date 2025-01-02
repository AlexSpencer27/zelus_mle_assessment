import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os

# Data folder
script_folder = Path(__file__).parent
data_folder = script_folder.parent / "data"


def main():
    # Load the trained model
    model_file = os.path.join(script_folder, "expected_runs_model.pkl")
    model = joblib.load(model_file)

    # Load the training dataset
    train_file = os.path.join(data_folder, "training_data.parquet")
    df = pd.read_parquet(train_file)

    # Features and target
    input_features = [
        "initial_batter",
        "initial_bowler",
        "num_batters",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    target = "runs"

    X = df[input_features]
    y_true = df[target]

    # Get predictions
    y_pred = model.predict(X)

    # Combine true and predicted values into a DataFrame
    results_df = pd.DataFrame({"True Runs": y_true, "Predicted Runs": y_pred})

    # Scatter plot using Plotly
    fig = px.scatter(
        results_df,
        x="True Runs",
        y="Predicted Runs",
        title="True vs. Predicted Runs",
        labels={"True Runs": "True Runs", "Predicted Runs": "Predicted Runs"},
        opacity=0.7,
    )

    # Add a reference line (y = x)
    fig.add_shape(
        type="line",
        x0=results_df["True Runs"].min(),
        y0=results_df["True Runs"].min(),
        x1=results_df["True Runs"].max(),
        y1=results_df["True Runs"].max(),
        line=dict(color="red", dash="dash"),
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    main()
