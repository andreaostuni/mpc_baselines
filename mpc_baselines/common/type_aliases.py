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

class MPCPolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        mpc_state: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict the action given an observation and an mpc state (and an optional hidden state)
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: The input observation
        :param mpc_state: The input mpc state
        :param state: The internal state of the model (used for RNNs)
        :param episode_start: Whether the current step is the start of a new episode
        :param deterministic: Whether to return deterministic action
        :return: The predicted action and the internal state of the model
        """
        ...