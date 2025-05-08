import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_path_json", type=Path)
    parser.add_argument("prediction_results_dir", type=Path)
    return parser.parse_args()


def calc_loss(inputs, prediction) -> float:
    ego_future = inputs["ego_agent_future"]  # (T, 3)
    ego_future = np.concatenate(
        [
            ego_future[..., :2],
            np.cos(ego_future[..., 2:3]),
            np.sin(ego_future[..., 2:3]),
        ],
        axis=-1,
    )  # (T, 4)
    neighbors_future = inputs["neighbor_agents_future"]  # (P32, T, 3)
    neighbor_future_mask = np.sum((neighbors_future[..., :3] != 0), axis=-1) == 0  # (P32, T)
    neighbors_future = np.concatenate(
        [
            neighbors_future[..., :2],
            np.cos(neighbors_future[..., 2:3]),
            np.sin(neighbors_future[..., 2:3]),
        ],
        axis=-1,
    )  # (P32, T, 4)
    neighbors_future[neighbor_future_mask] = 0.0

    P32, T, _ = neighbors_future.shape
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:4],
        inputs["neighbor_agents_past"][:P32, -1, :4],
    )
    # inputs = args.observation_normalizer(inputs)

    neighbor_current_mask = np.sum((neighbors_current[..., :4] != 0), axis=-1) == 0  # (P32)
    neighbor_mask = np.concatenate(
        (neighbor_current_mask[:, None], neighbor_future_mask), axis=-1
    )  # (P32, T + 1)
    neighbor_mask = neighbor_mask[:, -1]  # (P32) same to time axis

    gt_future = np.concatenate(
        [ego_future[None, :, :], neighbors_future[..., :]], axis=0
    )  # (1 + P32, T, 4)
    current_states = np.concatenate([ego_current[None, :], neighbors_current], axis=0)
    # (1 + P32, 4)

    all_gt = np.concatenate([current_states[:, None, :], gt_future], axis=1)  # (1 + P32, T + 1, 4)
    all_gt[1:][neighbor_mask] = 0.0  # (P32, T + 1, 4)

    pred_size = prediction.shape[0]

    neighbors_future_valid = ~neighbor_future_mask  # (P32, T)
    neighbors_future_valid = neighbors_future_valid[: (pred_size - 1)]  # (P10, T)
    all_gt = all_gt[:, 1:, :]  # (1 + P32, T, 4)
    all_gt = all_gt[:pred_size, :, :]  # (1 + P10, T, 4)

    loss_tensor = (prediction - all_gt) ** 2  # (1 + P10, T, 4)
    loss_ego = loss_tensor[0, :]  # (T, 4)
    loss_nei = loss_tensor[1:, :]  # (P10, T, 4)

    sum_neighbors_future_valid = np.sum(neighbors_future_valid, axis=1) > 0  # (P10)

    loss_nei = loss_nei[sum_neighbors_future_valid]  # (P_valid, T, 4)
    neighbors_future_valid = neighbors_future_valid[sum_neighbors_future_valid]  # (P_valid, T)

    return loss_ego, loss_nei, neighbors_future_valid


if __name__ == "__main__":
    args = parse_args()
    validation_path_json = args.validation_path_json
    prediction_results_dir = args.prediction_results_dir

    # Load the validation path JSON
    with open(validation_path_json, "r") as f:
        validation_paths = json.load(f)

    prediction_result_paths = sorted(prediction_results_dir.glob("*.npz"))

    assert len(validation_paths) == len(prediction_result_paths)

    ave_loss_ego = 0.0
    ave_loss_nei = 0.0
    num_loss_nei = 0

    for validation_path, prediction_result_path in zip(validation_paths, prediction_result_paths):
        validation_data = np.load(validation_path)
        prediction_result = np.load(prediction_result_path)["prediction"]

        loss_ego, loss_nei, neighbors_future_valid = calc_loss(validation_data, prediction_result)
        print(
            f"Validation Path: {validation_path}, "
            f"Prediction Result Path: {prediction_result_path}, "
            f"Loss Ego: {loss_ego.mean():.4f}, Loss Nei: {loss_nei.mean():.4f} {loss_nei.shape=}"
        )

        # ego
        loss_ego = loss_ego.mean()
        ave_loss_ego += loss_ego

        # neighbor
        if loss_nei.shape[0] > 0:
            curr_num = loss_nei.shape[0]
            loss_nei = loss_nei[neighbors_future_valid]
            loss_nei = loss_nei.mean() * curr_num
            ave_loss_nei += loss_nei
            num_loss_nei += curr_num

    ave_loss_ego /= len(validation_paths)
    ave_loss_nei /= num_loss_nei
    print(f"Average Loss Ego: {ave_loss_ego:.4f}, Average Loss Nei: {ave_loss_nei:.4f}")
