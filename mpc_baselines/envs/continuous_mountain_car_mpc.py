from ..wrappers.mpc_state import MPCStateWrapper
from ..wrappers.continuous_mountain_car_mpc import MountainCarContinuousMPC
import gymnasium as gym
import numpy as np

def MountainCarContinuousMPC_v0(**kwargs):
    return MountainCarContinuousMPC(**kwargs)