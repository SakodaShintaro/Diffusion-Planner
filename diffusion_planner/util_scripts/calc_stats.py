import argparse
import json
from collections import defaultdict
from pathlib import Path

import normalize
import numpy as np
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    limit = args.limit

    npz_path_list = sorted(root_dir.glob("**/*.npz"))

    print(f"Total {len(npz_path_list)} npz files")
    if limit > 0:
        div = len(npz_path_list) // limit
        npz_path_list = npz_path_list[::div]
    print(f"Check {len(npz_path_list)} npz files")

    keys = []
    stats = defaultdict(normalize.RunningStats)

    for npz_path in tqdm.tqdm(npz_path_list):
        data = np.load(npz_path, allow_pickle=True)

        for key in data.keys():
            if key in ["map_name", "token"]:
                continue
            val = data[key]

            if key == "ego_agent_future":
                stats[key].update(val.reshape(1, -1))
            elif key == "neighbor_agents_future":
                stats[key].update(val.reshape(val.shape[0], -1))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    norm_stats = {
        key: {"mean": value.mean.reshape(80, 3).tolist(), "std": value.std.reshape(80, 3).tolist()}
        for key, value in norm_stats.items()
    }

    output_path = "./norm_stats.json"
    json.dump(norm_stats, open(output_path, "w"), indent=2)
