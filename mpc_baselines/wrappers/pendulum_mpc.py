from .mpc_state import MPCStateWrapper
import numpy as np
class PendulumMPCWrapper(MPCStateWrapper):
    """
    Pendulum environment with MPC state.
    """

    def __init__(self, env = 'Pendulum-v1', **kwargs):
        super().__init__(env)

    def _get_mpc_state(self):
        return np.array([self.env.state.copy()[0], self.env.state.copy()[1]])
    