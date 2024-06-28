from ..wrappers.mpc_state import MPCStateWrapper
from ..wrappers.continuous_mountain_car_mpc import MountainCarContinuosMPC
import gymnasium as gym
import numpy as np

def MountainCarContinuousMPC_v1(**kwargs):
    return MountainCarContinuosMPC(**kwargs)