import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecMPCStateWrapper(VecEnvWrapper):
    """
    :param venv: The vectorized environment to wrap
    """

    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs
    
    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        return obs, rews, dones, infos

    def get_mpc_state(self) -> np.ndarray:
         return np.array(self.env_method("get_wrapper_attr", "mpc_state"))
