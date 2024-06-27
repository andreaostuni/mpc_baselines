from typing import Any, Dict, SupportsFloat, Tuple, Union
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

TimeFeatureObs = Union[np.ndarray, Dict[str, np.ndarray]]
MPCStateObs = Union[np.ndarray, Dict[str, np.ndarray]]


class MPCStateWrapper(gym.Wrapper):
    """
    Add a function to get the MPC state from the environment.
    .. note::

        Only ``gym.spaces.Box`` and ``gym.spaces.Dict`` (``gym.GoalEnv``) 1D observation spaces
        are supported for now.

    :param env: Gym env to wrap.
    """

    def __init__(self, env: gym.Env,**kwargs):
        super().__init__(env,**kwargs)

    @abstractmethod
    def _get_mpc_state(self) -> np.ndarray:
        # override this method to get the mpc state
        raise NotImplementedError
    
    @abstractmethod
    def _get_mpc_state_dim(self) -> int:
        # override this method to get the mpc state dimension
        raise NotImplementedError
    
    @property
    def mpc_state(self) -> np.ndarray:
        return self._get_mpc_state().astype(np.float32)
    
    @property
    def mpc_state_dim(self) -> int:
        return self._get_mpc_state_dim()