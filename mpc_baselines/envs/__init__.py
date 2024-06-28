from .pendulum_mpc import PendulumMPC_v1
from .continuous_mountain_car_mpc import MountainCarContinuousMPC_v1
import gymnasium as gym

__all__ = ['PendulumMPC_v1', 'MountainCarContinuousMPC_v1']

gym.envs.register(
    id='PendulumMPC-v1',
    entry_point='mpc_baselines.envs:PendulumMPC_v1',
    max_episode_steps=200
)

gym.envs.register(
  id='MountainCarContinuousMPC-v1',
  entry_point='mpc_baselines.envs:MountainCarContinuousMPC_v1',
    max_episode_steps=999
)