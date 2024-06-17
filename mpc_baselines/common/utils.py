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
from gymnasium import spaces

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
