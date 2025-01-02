import pandas as pd
from collections import defaultdict
from pathlib import Path
import tqdm
import os
from typing import Dict, List

# Define paths
script_folder: Path = Path(__file__).parent
data_folder: Path = script_folder.parent.parent / "data"
output_folder: Path = data_folder / "training"
os.makedirs(output_folder, exist_ok=True)


def main() -> None:
    """
    Main function to process filtered innings results, group by match, inning, and over,
    and generate training data with relevant features and targets. The output is saved
    as parquet files for downstream usage.
    """
    train_file: str = os.path.join(
        data_folder, "intermediate", "filtered_innings.parquet"
    )

    print("Reading filtered innings results")
    df: pd.DataFrame = pd.read_parquet(train_file)

    # Group data by matchid, innings, and over_int
    df_grouped = df.groupby(by=["matchid", "innings", "over_int"])

    # Dictionary to store training data
    train_dict: Dict[str, List] = defaultdict(list)

    # Initialize a progress bar
    pbar = tqdm.tqdm(total=len(df_grouped), desc="Creating training data")
    for (matchid, inning, over_num), group in df_grouped:
        # Extract metadata
        train_dict["matchid"].append(matchid)
        train_dict["date"].append(group["date"].iloc[0])
        train_dict["team"].append(group["team"].iloc[0])
        train_dict["opponent"].append(group["opponent"].iloc[0])
        train_dict["inning"].append(inning)
        train_dict["over_num"].append(over_num)

        # Extract features
        train_dict["initial_batter"].append(group["batsman_number"].iloc[0])
        train_dict["initial_bowler"].append(group["bowler_number"].iloc[0])
        train_dict["num_batsmen"].append(group["batsman_number"].nunique())
        train_dict["num_bowlers"].append(group["bowler_number"].nunique())
        train_dict["num_deliveries"].append(len(group))
        train_dict["remaining_wickets"].append(group["remaining_wickets"].min())
        train_dict["remaining_overs"].append(group["remaining_overs"].iloc[0])

        # Extract target
        train_dict["runs"].append(group["runs.total"].sum())

        pbar.update(1)
    pbar.close()

    # Convert to DataFrame
    print("Converting to DataFrame")
    train_df: pd.DataFrame = pd.DataFrame(train_dict)

    # Save training data to parquet files
    print("Writing to parquet")
    output_train_file: str = os.path.join(output_folder, "training_data.parquet")
    output_model_package_file: str = os.path.join(
        script_folder.parent, "model_package", "data.parquet"
    )
    train_df.to_parquet(output_train_file, index=False)
    train_df.to_parquet(output_model_package_file, index=False)

    print(
        f"Done. Training data saved to {output_train_file} and {output_model_package_file}"
    )


if __name__ == "__main__":
    main()
