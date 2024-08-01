import glob
import os
import platform
import random
import re
from collections import deque
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cloudpickle
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
import warnings 

import stable_baselines3 as sb3

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit

import stable_baselines3.common.utils as sb3_utils


def get_q_p_from_tensor(tensor: th.Tensor, timestamps: int) -> Tuple[th.Tensor, th.Tensor]:
    """
    Get the Q and p values for the MPC optimization from the output of the model.
    Q is a tensor of shape (timestamps, batch_size, nx + nu, nx + nu)
    p is a tensor of shape (timestamps, batch_size, nx + nu) 
    :param tensor: Output of the model of shape (batch_size, timestamps*(nx + nu)*2)
    :param timestamps: Number of timestamps
    :return: Q and p values
    """
    nx_plus_nu = tensor.shape[1] // (2 * timestamps)

    q = tensor[:, :timestamps * 2 * nx_plus_nu:2] # (batch_size, timestamps * (nx + nu))
    Q = q.view(-1, timestamps, nx_plus_nu).diag_embed() # (batch_size, timestamps, nx + nu, nx + nu)
    
    p = tensor[:, 1:timestamps * 2 * nx_plus_nu:2]
    p = p.view(-1, timestamps, nx_plus_nu)
    
    # exchange the first and second dimension
    Q = Q.permute(1, 0, 2, 3)
    p = p.permute(1, 0, 2)
    
    return Q, p


class ScaledSigmoid(nn.Module):
    r"""Applies the Sigmoid function element-wise and scales the output between an interval [a, b].

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
        \text{ScaledSigmoid}(x) =  

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ScaledSigmoid.png

    Examples::

        >>> m = nn.ScaledSigmoid(-2,2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    # def forward(self, input: th.Tensor) -> th.Tensor:
    #     return th.sigmoid(input)
    
    __constants__ = ['min_val', 'max_val']

    min_val: float
    max_val: float

    def __init__(
        self,
        min_val: float = 0.,
        max_val: float = 1.,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        super().__init__()
        if min_value is not None:
            warnings.warn(
                "keyword argument `min_value` is deprecated and rename to `min_val`",
                FutureWarning,
                stacklevel=2,
            )
            min_val = min_value
        if max_value is not None:
            warnings.warn(
                "keyword argument `max_value` is deprecated and rename to `max_val`",
                FutureWarning,
                stacklevel=2,
            )
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        assert self.max_val > self.min_val

    def forward(self, input: th.Tensor) -> th.Tensor:
        return th.sigmoid(input) * (self.max_val - self.min_val) + self.min_val

    def extra_repr(self) -> str:
        return f'min_val={self.min_val}, max_val={self.max_val}'