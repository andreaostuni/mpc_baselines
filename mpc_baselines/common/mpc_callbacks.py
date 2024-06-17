import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
import torch as th


try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

if TYPE_CHECKING:
    from stable_baselines3.common import base_class

from mpc.mpc import MPC
class MPCUpdateCallback(BaseCallback):
    """
    Callback for updating the MPC controller.

    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        """
        This method will be called by the model before each rollout starts.
        It is meant to retrieve the mpc state from the environment and update the MPC controller before the rollout.
        """
        # Retrieve the current environment
        env = self.training_env
        # Retrieve the MPC controller state from the environment
        mpc_state = env.get_attr("mpc_state")
        env_state = env.get_attr("state")
        # Update the MPC controller
        if isinstance(mpc_state, list):
            self.model.policy.update_mpc(*mpc_state)
            self.model.policy.update_env_state(*env_state)
        else:
            self.model.policy.update_mpc(mpc_state)
            self.model.policy.update_env_state(env_state)
        

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        It is meant to retrieve the mpc state from the environment and update the MPC controller on each step.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Retrieve the current environment
        env = self.training_env
        # Retrieve the MPC controller state from the environment
        mpc_state = env.get_attr("mpc_state")
        env_state = env.get_attr("state")
        print('time step', self.num_timesteps)
        
        if isinstance(mpc_state, list):
            self.model.policy.update_mpc(*mpc_state)
            self.model.policy.update_env_state(*env_state)
        else:
            self.model.policy.update_mpc(mpc_state)
            self.model.policy.update_env_state(env_state)
        return True