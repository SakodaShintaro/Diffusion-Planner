from torch.fx import symbolic_trace, GraphModule
from torch.fx.graph import Node
import torch.nn as nn
import torch
import copy

from diffusion_planner.model.module.encoder import *


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


class LaneFusionEncoderONNX(nn.Module):
    def __init__(self, original: LaneFusionEncoder):
        super().__init__()

        self._lane_len = original._lane_len
        self._channel = original._channel

        self.speed_limit_emb = original.speed_limit_emb
        self.unknown_speed_emb = original.unknown_speed_emb
        self.traffic_emb = original.traffic_emb

        self.channel_pre_project = original.channel_pre_project
        self.token_pre_project = original.token_pre_project

        self.blocks = original.blocks

        self.norm = original.norm
        self.emb_project = original.emb_project

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        valid_indices = ~mask_p.view(-1)

        x = x.view(B * P, V, -1)
        x = torch.where(
            valid_indices.view(-1, 1, 1),
            x,
            torch.zeros_like(x)
        )

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        speed_limit_input = speed_limit
        speed_limit_emb = self.speed_limit_emb(speed_limit_input)

        unknown_speed_emb = self.unknown_speed_emb(
            torch.zeros(B * P, dtype=torch.long, device=x.device))
        speed_limit_embedding = has_speed_limit * speed_limit_emb + \
            (~has_speed_limit) * unknown_speed_emb

        traffic_light_embedding = self.traffic_emb(traffic)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = x * valid_indices.float().unsqueeze(-1)
        return x_result.view(B, P, -1), mask_p.view(B, -1), pos.view(B, P, -1)


class StaticFusionEncoderONNX(nn.Module):
    def __init__(self, original: StaticFusionEncoder):
        super().__init__()
        self._hidden_dim = original._hidden_dim
        self.projection = original.projection  # Reuse original weights

    def forward(self, x):
        B, P, _ = x.shape

        # Create position tensor safely
        pos = x[:, :, :7]
        first_part = pos[..., :-3]
        middle = torch.zeros_like(pos[..., -3:])
        middle[..., -2] = 1.0
        pos = torch.cat([first_part, middle], dim=-1)

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1) == 0
        valid_mask = ~mask_p.view(-1)

        x_flat = x.view(B * P, -1)
        x_proj_input = x_flat[valid_mask]

        # Always do projection (ONNX can't branch)
        safe_input = torch.cat([
            x_proj_input,
            torch.zeros((1, x_flat.shape[-1]), device=x.device)
        ], dim=0)
        safe_output = self.projection(safe_input)

        # Take only the valid part (original number of valid rows)
        proj_out = safe_output[:x_proj_input.shape[0]]

        # Scatter result into x_result
        x_result[valid_mask] = proj_out

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class ONNXSafeModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = copy.deepcopy(original_model)
        self._replace_with_onnx_safe_modules()

    def _replace_with_onnx_safe_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, StaticFusionEncoder):
                parent_module = self._get_parent_module(name)
                subname = name.split(".")[-1]
                setattr(parent_module, subname,
                        StaticFusionEncoderONNX(module))
            if isinstance(module, LaneFusionEncoder):
                parent_module = self._get_parent_module(name)
                subname = name.split(".")[-1]
                setattr(parent_module, subname,
                        LaneFusionEncoderONNX(module))

    def _get_parent_module(self, name):
        parts = name.split(".")[:-1]
        mod = self.model
        for part in parts:
            mod = getattr(mod, part)
        return mod

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
