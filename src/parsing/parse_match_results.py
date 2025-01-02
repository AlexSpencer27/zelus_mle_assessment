import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import os
import warnings
import tqdm

warnings.filterwarnings("ignore")

script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"
output_folder = data_folder / "parsed"
os.makedirs(output_folder, exist_ok=True)


def main():
    match_results_file = os.path.join(
        data_folder, "provided_json", "match_results.json"
    )

    print(f"Reading from {match_results_file}")
    with open(match_results_file, "r") as f:
        match_results = json.load(f)

    key_columns = [
        "matchid",
        "match_type",
        "dates",
        "gender",
        "overs",
        "teams",
        "result",
        "outcome.wickets",
        "outcome.winner",
        "outcome.runs",
        "outcome.method",
    ]

    pbar = tqdm.tqdm(total=len(match_results), desc="Parsing match results")

    dict_results = defaultdict(list)
    for match in match_results:
        for key in key_columns:
            dict_results[key].append(match.get(key, None))

        pbar.update(1)
    pbar.close()

    print("Converting to DataFrame")
    df = pd.DataFrame(dict_results)

    print("Converting to parquet")
    df.to_parquet(
        os.path.join(output_folder, "match_results.parquet"), index=False
    )
    print("Done")


if __name__ == "__main__":
    main()
