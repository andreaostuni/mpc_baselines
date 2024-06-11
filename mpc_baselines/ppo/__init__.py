# from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
# from stable_baselines3.ppo.ppo import PPO
from mpc_baselines.ppo.policies import MPCMlpPolicy, MPCCnnPolicy, MPCMultiInputPolicy

__all__ = ['MPCMlpPolicy', 'MPCCnnPolicy', 'MPCMultiInputPolicy']
