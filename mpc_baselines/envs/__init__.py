from .pendulum_mpc import PendulumMPC_v1
import gymnasium as gym

__all__ = ['PendulumMPC_v1']

gym.envs.register(
    id='PendulumMPC-v1',
    entry_point='mpc_baselines.envs:PendulumMPC_v1',
    max_episode_steps=200
)