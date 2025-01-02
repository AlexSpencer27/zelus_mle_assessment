import pandas as pd
from collections import defaultdict
from pathlib import Path
import tqdm
import os

# data folder is one up from this script
script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"

output_folder = data_folder / "training"
os.makedirs(output_folder, exist_ok=True)


def main():
    train_file = os.path.join(data_folder, "intermediate", "filtered_innings.parquet")

    print("Reading filtered innings results")
    df = pd.read_parquet(train_file)

    df_grouped = df.groupby(by=["matchid", "innings", "over_int"])

    train_dict = defaultdict(list)

    pbar = tqdm.tqdm(total=len(df_grouped), desc="Creating training data")
    for (matchid, inning, over_num), group in df_grouped:
        # extract metadata
        train_dict["matchid"].append(matchid)
        train_dict["date"].append(group["date"].iloc[0])
        train_dict["team"].append(group["team"].iloc[0])
        train_dict["opponent"].append(group["opponent"].iloc[0])
        train_dict["inning"].append(inning)
        train_dict["over_num"].append(over_num)

        # extract features
        train_dict["initial_batter"].append(group["batsman_number"].iloc[0])
        train_dict["initial_bowler"].append(group["bowler_number"].iloc[0])
        train_dict["num_batsmen"].append(group["batsman_number"].nunique())
        train_dict["num_bowlers"].append(group["bowler_number"].nunique())
        train_dict["num_deliveries"].append(len(group))
        train_dict["remaining_wickets"].append(group["remaining_wickets"].min())
        train_dict["remaining_overs"].append(group["remaining_overs"].iloc[0])

        # extract target
        train_dict["runs"].append(group["runs.total"].sum())

        pbar.update(1)
    pbar.close()

    print("Converting to DataFrame")
    train_df = pd.DataFrame(train_dict)

    print("Writing to parquet")
    train_df.to_parquet(
        os.path.join(output_folder, "training_data.parquet"), index=False
    )
    train_df.to_parquet(
        os.path.join(script_folder.parent, "model_package", "data.parquet"),
        index=False,
    )
    print("Done")


if __name__ == "__main__":
    main()
