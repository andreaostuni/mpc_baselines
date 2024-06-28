from .mpc_state import MPCStateWrapper
import gymnasium as gym
import numpy as np

class MountainCarContinuousMPC(MPCStateWrapper):
    """
    MountainCarContinuous environment with MPCStateWrapper
    """

    def __init__(self, env = 'MountainCarContinuous-v0', **kwargs):
        if isinstance(env, str):
            env = gym.make(env,**kwargs).unwrapped
        super().__init__(env, **kwargs)
        
    def _get_mpc_state(self):
        return np.array([self.env.unwrapped.state.copy()[0], self.env.unwrapped.state.copy()[1]])
    
    def _get_mpc_state_dim(self):
        return 2