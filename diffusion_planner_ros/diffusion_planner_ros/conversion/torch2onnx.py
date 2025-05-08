import argparse
import json
import random

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx_compatible_transformer import *

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

torch.backends.mha.set_fastpath_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert torch model to onnx")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="If set, only test the output of the existing ONNX model",
    )

    parser.add_argument("--config", type=str, default="args.json", help="Config file path")
    parser.add_argument("--ckpt", type=str, default="latest.pth", help="Checkpoint file path")
    parser.add_argument("--onnx_path", type=str, default="model.onnx", help="ONNX model file path")
    parser.add_argument(
        "--sample_input_path", type=str, default="sample_input.npz", help="Sample input path"
    )
    parser.add_argument(
        "--wrap_with_onnx_functions",
        action="store_true",
        help="Wether to replace some original functions with onnx-friendly ones",
    )
    args = parser.parse_args()
    return args


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        ego_current_state,
        neighbor_agents_past,
        static_objects,
        lanes,
        lanes_speed_limit,
        lanes_has_speed_limit,
        route_lanes,
    ):
        inputs = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
        }
        encoder_outputs, decoder_outputs = self.model(inputs)
        return decoder_outputs


def create_input(input_dict, input_type="onnx"):
    if input_type == "onnx":
        return {
            "ego_current_state": input_dict["ego_current_state"].cpu().numpy(),
            "neighbor_agents_past": input_dict["neighbor_agents_past"].cpu().numpy(),
            "static_objects": input_dict["static_objects"].cpu().numpy(),
            "lanes": input_dict["lanes"].cpu().numpy(),
            "lanes_speed_limit": input_dict["lanes_speed_limit"].cpu().numpy(),
            "lanes_has_speed_limit": input_dict["lanes_has_speed_limit"].cpu().numpy(),
            "route_lanes": input_dict["route_lanes"].cpu().numpy(),
        }
    elif input_type == "torch":
        return (
            input_dict["ego_current_state"],
            input_dict["neighbor_agents_past"],
            input_dict["static_objects"],
            input_dict["lanes"],
            input_dict["lanes_speed_limit"],
            input_dict["lanes_has_speed_limit"],
            input_dict["route_lanes"],
        )


def compare_outputs(torch_output, onnx_output):
    print(f"torch output, with shape {torch_output.shape}:")
    print(f"onnx output, with shape {onnx_output[0].shape}:")
    abs_diff = np.abs(torch_output - onnx_output[0])
    print(f"Max diff: {abs_diff.max()}")
    print(f"Mean diff: {abs_diff.mean()}")
    print(f"Close? {np.allclose(torch_output, onnx_output[0], rtol=1e-03, atol=1e-05)}")


if __name__ == "__main__":
    args = parse_args()
    config_json_path = args.config
    ckpt_path = args.ckpt
    onnx_path = args.onnx_path
    sample_input_path = args.sample_input_path
    wrap_with_onnx = args.wrap_with_onnx_functions
    test_only = args.test_only

    # Load config
    with open(config_json_path, "r") as f:
        config_json = json.load(f)
    config_obj = Config(config_json_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # # Dummy inputs
    # dummy_inputs = {
    #     'ego_current_state': torch.tensor(np.random.randn(10).astype(np.float32)).unsqueeze(0),
    #     'neighbor_agents_past': torch.tensor(np.random.randn(32, 21, 11).astype(np.float32)).unsqueeze(0),
    #     'static_objects': torch.tensor(np.random.randn(5, 10).astype(np.float32)).unsqueeze(0),
    #     'lanes': torch.tensor(np.random.randn(70, 20, 12).astype(np.float32)).unsqueeze(0),
    #     'lanes_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.float32)).unsqueeze(0),
    #     'lanes_has_speed_limit': torch.tensor(np.random.randn(70, 1).astype(np.bool_)).unsqueeze(0),
    #     'route_lanes': torch.tensor(np.random.randn(25, 20, 12).astype(np.float32)).unsqueeze(0),
    # }

    dummy_inputs = {
        "ego_current_state": torch.full((1, 10), 0.5, dtype=torch.float32),
        "neighbor_agents_past": torch.full((1, 32, 21, 11), 0.5, dtype=torch.float32),
        "static_objects": torch.full((1, 5, 10), 0.5, dtype=torch.float32),
        "lanes": torch.full((1, 70, 20, 12), 0.5, dtype=torch.float32),
        "lanes_speed_limit": torch.full((1, 70, 1), 0.5, dtype=torch.float32),
        "lanes_has_speed_limit": torch.full((1, 70, 1), True, dtype=torch.bool),
        "route_lanes": torch.full((1, 25, 20, 12), 0.5, dtype=torch.float32),
    }

    input_names = [
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
        "route_lanes",
    ]

    # Export
    # Init model
    model = Diffusion_Planner(config_obj)
    model.eval()
    model.encoder.encoder.eval()
    model.decoder.decoder.eval()
    model.decoder.decoder.training = False

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Wrap model for onnx compatibility
    if wrap_with_onnx:
        onnx_safe_model = ONNXSafeModel(model).eval()
        wrapper = ONNXWrapper(onnx_safe_model).eval()
    else:
        wrapper = ONNXWrapper(model).eval()

    # Prepare input
    torch_input_tuple = create_input(dummy_inputs, "torch")

    if not test_only:
        print(f"creating a new onnx model: {onnx_path}")
        onnx_model = torch.onnx.export(
            wrapper,
            torch_input_tuple,
            onnx_path,
            input_names=input_names,
            output_names=["output"],
            # dynamic_axes=None,
            dynamic_axes={name: {0: "batch"} for name in input_names},  # optional, but useful
            opset_version=20,
        )

    with torch.no_grad():
        output = wrapper(*torch_input_tuple)
        torch_output = output["prediction"].cpu().numpy()

    sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort_session = ort.InferenceSession(
        onnx_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    onnx_inputs = create_input(dummy_inputs, "onnx")
    onnx_output = ort_session.run(None, onnx_inputs)
    print("Compare outputs using the creation input")
    compare_outputs(torch_output, onnx_output)

    # TEST WITH SAMPLE INPUT
    # Load the sample input
    sample_input_file = np.load(sample_input_path)
    sample_input = {}
    for key in input_names:
        sample_input[key] = torch.tensor(sample_input_file[key]).unsqueeze(0)

    sample_input = config_obj.observation_normalizer(sample_input)

    # Run torch inference
    torch_input_tuple = create_input(sample_input, "torch")
    with torch.no_grad():
        output = wrapper(*torch_input_tuple)
        torch_output = output["prediction"].cpu().numpy()
    onnx_inputs = create_input(sample_input, "onnx")
    onnx_output = ort_session.run(None, onnx_inputs)

    print("Compare outputs using a sample input")
    compare_outputs(torch_output, onnx_output)
