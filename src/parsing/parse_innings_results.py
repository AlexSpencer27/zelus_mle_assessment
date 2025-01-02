import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import os
import tqdm
import warnings
from typing import List, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define paths
script_folder: Path = Path(__file__).parent
data_folder: Path = script_folder.parent.parent / "data"
output_folder: Path = data_folder / "parsed"
os.makedirs(output_folder, exist_ok=True)


def main() -> None:
    """
    Main function to parse innings results from a JSON file,
    transform them into a structured DataFrame, and save the output as a parquet file.
    """
    innings_results_file: str = os.path.join(
        data_folder, "provided_json", "innings_results.json"
    )

    print(f"Reading from {innings_results_file}")
    with open(innings_results_file, "r") as f:
        innings_results: List[Dict] = json.load(f)

    key_columns: List[str] = [
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

    # Initialize a progress bar
    pbar = tqdm.tqdm(total=len(innings_results), desc="Parsing innings results")

    # Initialize a dictionary to hold parsed results
    dict_results: Dict[str, List] = defaultdict(list)
    for innings in innings_results:
        for key in key_columns:
            dict_results[key].append(innings.get(key, None))
        pbar.update(1)
    pbar.close()

    print("Converting to DataFrame")
    df: pd.DataFrame = pd.DataFrame(dict_results)

    print("Saving to parquet")
    output_file: str = os.path.join(output_folder, "innings_results.parquet")
    df.to_parquet(output_file, index=False)
    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    main()
