from ..wrappers.mpc_state import MPCStateWrapper
import gymnasium as gym
import numpy as np
class PendulumMPCWrapper(MPCStateWrapper):
    """
    Pendulum environment with MPC state.
    """
    def angle_normalize(self, x):
        ''' Normalize the angle to be in [-pi, pi] '''
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def __init__(self, env = 'Pendulum-v1', **kwargs):
        if isinstance(env, str):
            env = gym.make(env).unwrapped
        super().__init__(env, **kwargs)
        
    def _get_mpc_state(self):
        return np.array([self.angle_normalize(self.env.unwrapped.state.copy()[0]), self.env.unwrapped.state.copy()[1]])
    
def PendulumMPC_v1(**kwargs):
    return PendulumMPCWrapper(**kwargs)