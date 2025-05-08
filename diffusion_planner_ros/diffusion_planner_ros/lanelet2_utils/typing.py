from __future__ import annotations

from logging import RootLogger
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.utils.data as torch_data
from numpy.typing import ArrayLike as _ArrayLike
from numpy.typing import NDArray

# from numpy.typing._shape import _Shape
from torch import nn

# === torch ===
_TorchDType = TypeVar("_TorchDType", bound=torch.dtype)


class _Tensor(torch.Tensor, Generic[_TorchDType]):
    pass


Tensor = _Tensor[torch.dtype]
TensorF64 = _Tensor[torch.float64]
TensorF32 = _Tensor[torch.float32]
TensorI64 = _Tensor[torch.int64]
TensorI32 = _Tensor[torch.int32]
TensorBool = _Tensor[torch.bool]

DeviceLike = str | torch.device | int

Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa

Dataset = torch_data.Dataset  # BaseDataset
DataLoader = torch_data.DataLoader

Module = nn.Module
ModuleList = nn.ModuleList
ModuleDict = nn.ModuleDict
SequentialModule = nn.Sequential
Parameter = nn.Parameter
ParameterList = nn.ParameterList
ParameterDict = nn.ParameterDict

Graph = torch.Graph
# GraphCtx: TypeAlias = torch.onnx._internal.jit_utils.GraphContext  # noqa: SLF001
GraphCtx = "GraphCtx"
JitValue = torch.Value

# === numpy ===
NDArrayFloat = NDArray[np.float64 | np.float32]
NDArrayF64 = NDArray[np.float64]
NDArrayF32 = NDArray[np.float32]
NDArrayInt = NDArray[np.int64 | np.float32]
NDArrayI64 = NDArray[np.int64]
NDArrayI32 = NDArray[np.int32]
NDArrayBool = NDArray[np.bool_]
NDArrayStr = NDArray[np.str_]

ArrayLike = _ArrayLike
# ArrayShape = _Shape
ArrayShape = NDArray.shape


# logging
Logger = RootLogger
