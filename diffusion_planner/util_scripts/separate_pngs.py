import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir

    png_path_list = sorted(input_dir.glob("*.png"))
    result_dir = input_dir.parent / "visualization_separated"
    result_dir.mkdir(exist_ok=True)

    for png_path in png_path_list:
        time = png_path.stem.split("_")[0]
        time_dir = result_dir / time
        time_dir.mkdir(exist_ok=True)
        shutil.copy(png_path, time_dir)
