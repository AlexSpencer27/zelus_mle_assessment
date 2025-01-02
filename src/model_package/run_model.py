import pandas as pd
import joblib
import typer
from pathlib import Path
import os
import logging

# Initialize Typer app
app = typer.Typer()

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

script_folder = Path(__file__).parent


@app.command()
def main(
    model: str = typer.Option(
        os.path.join(script_folder, "expected_runs_model.pkl"),
        help="Path to the trained model file",
    ),
    data: str = typer.Option(
        os.path.join(script_folder, "data.parquet"),
        help="Path to the input data file",
    ),
    batting_team: str = typer.Option(None, help="Batting team to filter by"),
    bowling_team: str = typer.Option("None", help="Bowling team to filter by"),
    start_over: int = typer.Option(1, help="Start of over range (inclusive)"),
    end_over: int = typer.Option(5, help="End of over range (inclusive)"),
    num_matches: int = typer.Option(1, help="Number of most recent matches to use"),
    match_order: str = typer.Option("oldest", help="Order of matches to use"),
):
    """
    Run predictions for cricket overs.
    """
    # Interactive input if team not provided
    if not batting_team:
        batting_team = "Ireland"
        logging.info("No batting team provided - using Ireland")

    # Log arguments
    logging.info(
        f"Arguments:\nModel: {model}\nData: {data}\nBatting Team: {batting_team}\nBowling Team: {bowling_team}\n"
        f"Start Over: {start_over}\nEnd Over: {end_over}\nNum Matches: {num_matches}"
    )

    # Load and filter data
    logging.info(f"Loading data from {data}")
    data_filtered = load_data(
        data_path=data,
        batting_team=batting_team,
        bowling_team=bowling_team,
        start_over=start_over,
        end_over=end_over,
        num_matches=num_matches,
        match_order=match_order,
    )

    # Load the model
    logging.info(f"Loading model from {model}")
    model_obj = load_model(model)

    # Input features
    input_features = [
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    X = data_filtered[input_features]

    # Make predictions
    predictions = model_obj.predict(X)

    # Display predictions
    result = pd.DataFrame(
        {
            "matchid": data_filtered["matchid"],
            "date": data_filtered.get("date", pd.NA),
            "batting_team": data_filtered["team"],
            "bowling_team": data_filtered["opponent"],
            "over_num": data_filtered["over_num"],
            "predicted_runs": predictions,
        }
    )

    typer.echo("\n" + result.to_string(index=False))


def load_model(model_path: str):
    """
    Load the trained model. Raises a ValueError if the model path is invalid.
    """
    if not os.path.exists(model_path):
        raise typer.BadParameter(
            f"Invalid model path: {model_path}. Please provide a valid path."
        )

    try:
        return joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}. Error: {e}")


def load_data(
    data_path: str,
    batting_team: str,
    bowling_team: str,
    start_over: int,
    end_over: int,
    num_matches: int,
    match_order: str,
) -> pd.DataFrame:
    """
    Load and filter data based on the input parameters.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the dataset is empty or missing critical columns.
    """
    # Validate data file existence
    if not os.path.exists(data_path):
        raise typer.BadParameter(
            f"Invalid data file path: {data_path}. Please provide a valid path."
        )

    try:
        # Load the dataset
        df = pd.read_parquet(data_path)
    except Exception as e:
        raise typer.BadParameter(
            f"Failed to load data from {data_path}. Error: {str(e)}"
        )

    # Validate required columns
    required_columns = [
        "matchid",
        "team",
        "opponent",
        "over_num",
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise typer.BadParameter(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    # Apply filters
    try:
        df = validate_and_filter_team(df, batting_team)
        df = validate_and_filter_opponent(df, bowling_team)
        df = filter_by_overs(df, start_over, end_over)
        df = filter_by_recent_matches(df, num_matches, match_order)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # Validate non-empty dataset after filtering
    if df.empty:
        raise typer.BadParameter(
            "No data available after applying filters. Please check your inputs."
        )

    return df


def validate_and_filter_team(df: pd.DataFrame, filter_team: str) -> pd.DataFrame:
    """
    Validate the team and filter the dataset.
    """
    valid_teams = df.team.unique()
    if filter_team.lower() not in map(str.lower, valid_teams):
        valid_teams_str = ", ".join(sorted(valid_teams))
        raise typer.BadParameter(
            f"Batting team '{filter_team}' not found. Please choose from: {valid_teams_str}"
        )
    return df.loc[df["team"].str.lower() == filter_team.lower()]


def validate_and_filter_opponent(df: pd.DataFrame, bowling_team: str) -> pd.DataFrame:
    """
    Validate the opponent and filter the dataset.
    """
    if bowling_team == "None":
        return df

    valid_opponents = df.opponent.unique()
    if bowling_team.lower() not in map(str.lower, valid_opponents):
        valid_opponents_str = ", ".join(sorted(valid_opponents))
        raise typer.BadParameter(
            f"Bowling team '{bowling_team}' never played {df.iloc[0]['team']}. Please choose from: {valid_opponents_str}"
        )
    return df.loc[df["opponent"].str.lower() == bowling_team.lower()]


def filter_by_overs(df: pd.DataFrame, start_over: int, end_over: int) -> pd.DataFrame:
    """
    Filter the dataset by the specified over range.
    """
    if start_over > end_over:
        raise typer.BadParameter("Start over must be less than or equal to end over")
    if start_over < 1:
        logging.warning("Start over cannot be less than 1. Adjusting to 1.")
        start_over = 1
    if end_over > 50:
        logging.warning("End over cannot be greater than 50. Adjusting to 50.")
        end_over = 50

    return df.loc[(df["over_num"] >= start_over) & (df["over_num"] <= end_over)]


def filter_by_recent_matches(
    df: pd.DataFrame, num_matches: int, match_order: str
) -> pd.DataFrame:
    """
    Filter the dataset to the most recent matches.
    """
    if num_matches == -1:
        return df

    if num_matches < 1:
        raise typer.BadParameter("Number of matches must be at least 1.")
    unique_matches = df["matchid"].unique()
    if len(unique_matches) >= num_matches:

        if match_order not in ["oldest", "newest"]:
            raise typer.BadParameter(
                f"match-order must be oldest or newest, not {match_order}"
            )

        asc = match_order == "oldest"

        most_recent_matches = df.sort_values("date", ascending=asc)["matchid"].unique()[
            :num_matches
        ]

        df = df[df["matchid"].isin(most_recent_matches)]
        logging.info(f"Filtered to the most recent {num_matches} matches.")
    return df


if __name__ == "__main__":
    app()
