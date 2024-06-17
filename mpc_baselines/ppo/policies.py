# This file is here just to define MPCMlpPolicy/MPCCnnPolicy
# that work for PPO
from mpc_baselines.common.policies import MPCActorCriticCnnPolicy, MPCActorCriticPolicy, MPCMultiInputActorCriticPolicy

MPCMlpPolicy = MPCActorCriticPolicy
MPCCnnPolicy = MPCActorCriticCnnPolicy
MPCMultiInputPolicy = MPCMultiInputActorCriticPolicy