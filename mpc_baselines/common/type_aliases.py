"""Common aliases for type hints"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

import stable_baselines3.common.type_aliases as sb3_types

class MPCRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    mpc_states : th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class MPCDictRolloutBufferSamples(NamedTuple):
    observations: sb3_types.TensorDict
    actions: th.Tensor
    mpc_states : th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class MPCReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    mpc_states : th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class MPCDictReplayBufferSamples(NamedTuple):
    observations: sb3_types.TensorDict
    actions: th.Tensor
    mpc_states : th.Tensor
    next_observations: sb3_types.TensorDict
    dones: th.Tensor
    rewards: th.Tensor