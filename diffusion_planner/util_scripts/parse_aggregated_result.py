import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregator_metric_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregator_metric_dir = args.aggregator_metric_dir

    files = sorted(aggregator_metric_dir.glob("*.parquet"))
    assert len(files) == 1
    f = files[0]
    df = pd.read_parquet(f)
    name = os.path.splitext(os.path.basename(f))[0]
    print(f"{len(df)=}")

    # log_nameがNoneでない行だけ残す
    df = df[df["log_name"].notnull()]
    print(f"{len(df)=}")

    mean = df["score"].mean()
    print(f"{mean=}")
