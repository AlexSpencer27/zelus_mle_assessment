import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import os
import tqdm
import warnings

warnings.filterwarnings("ignore")

script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"
output_folder = data_folder / "parsed"
os.makedirs(output_folder, exist_ok=True)


def main():
    innings_results_file = os.path.join(
        data_folder, "provided_json", "innings_results.json"
    )

    print(f"Reading from {innings_results_file}")
    with open(innings_results_file, "r") as f:
        innings_results = json.load(f)

    key_columns = [
        "batsman",
        "bowler",
        "over",
        "team",
        "innings",
        "matchid",
        "wicket.kind",
        "runs.batsman",
        "runs.extras",
        "runs.total",
    ]

    pbar = tqdm.tqdm(total=len(innings_results), desc="Parsing innings results")
    dict_results = defaultdict(list)
    for innings in innings_results:
        for key in key_columns:
            dict_results[key].append(innings.get(key, None))

        pbar.update(1)
    pbar.close()

    print("Converting to DataFrame")
    df = pd.DataFrame(dict_results)

    print("Converting to parquet")
    df.to_parquet(
        os.path.join(output_folder, "innings_results.parquet"), index=False
    )
    print("Done")


if __name__ == "__main__":
    main()
