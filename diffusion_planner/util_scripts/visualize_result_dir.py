import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    return parser.parse_args()


def load_mean_metrics(metrics_dir):
    files = sorted(metrics_dir.glob("*.parquet"))
    mean_metrics = {}
    for f in files:
        df = pd.read_parquet(f)
        name = os.path.splitext(os.path.basename(f))[0]
        if df["metric_score"].mean() > 0:
            mean_metrics[name] = df["metric_score"].mean()
    return mean_metrics


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir

    target_dir_list = sorted(root_dir.glob("*/"))
    metrics = []
    for target_dir in target_dir_list:
        try:
            elements = str(target_dir.name).split("_")
            print(elements)
            epoch = int(elements[2])
        except:  # noqa: E722
            continue
        m = load_mean_metrics(target_dir / "metrics")
        m["epoch"] = epoch
        metrics.append(m)

    save_dir = root_dir / "metric_graphs_dir"
    save_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(metrics)
    df = df.set_index("epoch")
    df.to_csv(save_dir / "result.csv")

    for col in df.columns:
        if col == "epoch":
            continue

        plt.figure(figsize=(8, 6))
        plt.plot(df["epoch"], df[col])
        plt.xlabel("epoch")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(save_dir / f"{col}.png", bbox_inches="tight", pad_inches=0.1)
        plt.close()
