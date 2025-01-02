import pandas as pd
from pathlib import Path
import os

# Define paths
script_folder: Path = Path(__file__).parent
data_folder: Path = script_folder.parent.parent / "data"
output_folder: Path = data_folder / "intermediate"
os.makedirs(output_folder, exist_ok=True)


def main() -> None:
    """
    Reads filtered innings results, extracts key columns, and saves the output as a CSV file
    for further analysis.
    """
    filtered_innings_file: str = os.path.join(
        data_folder, "intermediate", "filtered_innings.parquet"
    )

    print("Reading filtered innings results")
    df: pd.DataFrame = pd.read_parquet(filtered_innings_file)

    # Define key columns for the output
    key_cols = {
        "matchid": {"dtp": int, "rename": "match_id"},
        "date": {"dtp": str, "rename": "date"},
        "team": {"dtp": str, "rename": "batting_team"},
        "opponent": {"dtp": str, "rename": "bowling_team"},
        "innings": {"dtp": int, "rename": "innings_order"},
        "remaining_overs": {"dtp": int, "rename": "remaining_overs"},
        "remaining_wickets": {"dtp": int, "rename": "remaining_wickets"},
    }

    print("Taking subset of columns")
    df_out: pd.DataFrame = pd.DataFrame()

    for key, val in key_cols.items():
        df_out[val["rename"]] = df[key].astype(val["dtp"])

    output_file: str = os.path.join(output_folder, "q3a.csv")

    print("Writing to CSV")
    df_out.to_csv(output_file, index=False)
    print(f"Done. Output saved to {output_file}")


if __name__ == "__main__":
    main()
