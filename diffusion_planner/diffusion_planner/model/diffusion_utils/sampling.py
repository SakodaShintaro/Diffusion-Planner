from typing import Dict

import torch

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


@torch.no_grad()
def dpm_sampler(
    model: torch.nn.Module,
    x_T: torch.Tensor,
    other_model_params: Dict,
    model_wrapper_params: Dict,
    dpm_solver_params: Dict,
):
    noise_schedule = dpm.NoiseScheduleVP()

    model_fn = dpm.model_wrapper(
        model,
        noise_schedule,
        model_type=model.model_type,
        model_kwargs=other_model_params,
        **model_wrapper_params,
    )

    dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, **dpm_solver_params)

    sample_dpm = dpm_solver.sample(x_T, steps=10, skip_type="logSNR")

    return sample_dpm
