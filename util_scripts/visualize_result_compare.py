import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir1", type=Path)
    parser.add_argument("metrics_dir2", type=Path)
    parser.add_argument("label1", type=Path)
    parser.add_argument("label2", type=Path)
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
    m1 = load_mean_metrics(args.metrics_dir1 / "metrics")
    m2 = load_mean_metrics(args.metrics_dir2 / "metrics")

    # 共通指標に揃える
    all_keys = sorted(set(m1.keys()) | set(m2.keys()))
    v1 = [m1.get(k, 0) for k in all_keys]
    v2 = [m2.get(k, 0) for k in all_keys]

    y = np.arange(len(all_keys))
    h = 0.4

    plt.figure(figsize=(8, 6))
    plt.barh(y + h / 2, v1, height=h, label=args.label1)
    plt.barh(y - h / 2, v2, height=h, label=args.label2)
    plt.yticks(y, all_keys)
    plt.xlabel("metric_score")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    save_dir = args.metrics_dir1 / "metric_graphs_compare"
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "compare.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
