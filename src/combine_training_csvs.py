import argparse
from pathlib import Path

import pandas as pd


def load_with_weight(path, weight, source_name):
    df = pd.read_csv(path, low_memory=False)
    df["sample_weight"] = weight
    df["training_source"] = source_name
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine the primary temporal CSV with auxiliary datasets."
    )
    parser.add_argument(
        "--primary-csv",
        default="data/features_video_temporal_clear_binary.csv",
    )
    parser.add_argument(
        "--aux-csv",
        default="data/features_yawdd_temporal_aux.csv",
    )
    parser.add_argument(
        "--output",
        default="data/features_temporal_with_yawdd.csv",
    )
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    primary_df = load_with_weight(args.primary_csv, args.primary_weight, "primary_video")
    aux_df = load_with_weight(args.aux_csv, args.aux_weight, "yawdd_auxiliary")

    all_columns = list(dict.fromkeys([*primary_df.columns, *aux_df.columns]))
    primary_df = primary_df.reindex(columns=all_columns)
    aux_df = aux_df.reindex(columns=all_columns)
    combined_df = pd.concat([primary_df, aux_df], ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Primary rows: {len(primary_df)}")
    print(f"Aux rows:     {len(aux_df)}")
    print(f"Total rows:   {len(combined_df)}")
    print("Rows by source:")
    print(combined_df["training_source"].value_counts().to_string())
    print("Effective weight by source:")
    print(combined_df.groupby("training_source")["sample_weight"].sum().to_string())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
