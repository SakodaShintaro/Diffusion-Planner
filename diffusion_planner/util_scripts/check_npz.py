import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npz_path = args.npz_path

    # Load the npz file
    data = np.load(npz_path, allow_pickle=True)

    # Print the shape of each array
    for key in data.keys():
        val = data[key]
        print(f"{key}\t{val.dtype}\t{val.shape}")
