import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_dir = args.metrics_dir

    files = sorted(metrics_dir.glob("*.parquet"))
    metrics = {}
    mean_metrics = {}
    for f in files:
        df = pd.read_parquet(f)
        name = os.path.splitext(os.path.basename(f))[0]
        metrics[name] = df
        if df["metric_score"].mean() > 0:
            mean_metrics[name] = df["metric_score"].mean()

    print(mean_metrics)

    # Plot "metric_score"
    save_dir = metrics_dir.parent / "metric_graphs"
    save_dir.mkdir(exist_ok=True)
    plt.barh(mean_metrics.keys(), mean_metrics.values())
    plt.tight_layout()
    plt.savefig(save_dir / "mean.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
