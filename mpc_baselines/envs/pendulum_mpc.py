from ..wrappers.mpc_state import MPCStateWrapper
from ..wrappers.pendulum_mpc import PendulumMPCWrapper
import gymnasium as gym
import numpy as np

def PendulumMPC_v1(**kwargs):
    return PendulumMPCWrapper(**kwargs)