"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_action_dim
# from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution, sum_independent_dims

# SelfDistribution = TypeVar("SelfDistribution", bound="Distribution")
SelfMPCDiagGaussianDistribution = TypeVar("SelfMPCDiagGaussianDistribution", bound="MPCDiagGaussianDistribution")
SelfMPCSquashedDiagGaussianDistribution = TypeVar(
    "SelfMPCSquashedDiagGaussianDistribution", bound="MPCSquashedDiagGaussianDistribution"
)
# SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")
# SelfMultiCategoricalDistribution = TypeVar("SelfMultiCategoricalDistribution", bound="MultiCategoricalDistribution")
# SelfBernoulliDistribution = TypeVar("SelfBernoulliDistribution", bound="BernoulliDistribution")
# SelfStateDependentNoiseDistribution = TypeVar("SelfStateDependentNoiseDistribution", bound="StateDependentNoiseDistribution")

import stable_baselines3.common.distributions as sb3_distributions 

class MPCDiagGaussianDistribution(sb3_distributions.DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, mpc_horizon: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.mpc_horizon = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        QP = nn.Linear(latent_dim, 2* self.action_dim * self.mpc_horizon)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return QP, log_std

    def proba_distribution(
        self: SelfMPCDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfMPCDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sb3_distributions.sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return sb3_distributions.sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MPCSquashedDiagGaussianDistribution(MPCDiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int,  mpc_horizon: int, epsilon: float = 1e-6):
        super().__init__(action_dim, mpc_horizon)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions: Optional[th.Tensor] = None

    def proba_distribution(
        self: SelfMPCSquashedDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfMPCSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = sb3_distributions.TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


def make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, use_mpc = True, dist_kwargs: Optional[Dict[str, Any]] = None
) -> sb3_distributions.Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if use_mpc:
        if isinstance(action_space, spaces.Box):
            cls = MPCDiagGaussianDistribution
            return cls(get_action_dim(action_space), **dist_kwargs)
    else:
        sb3_distributions.make_proba_distribution(action_space, use_sde, dist_kwargs)


def kl_divergence(dist_true: sb3_distributions.Distribution, dist_pred: sb3_distributions.Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence

    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert isinstance(dist_true, MultiCategoricalDistribution)  # already checked above, for mypy
        assert np.allclose(
            dist_pred.action_dims, dist_true.action_dims
        ), f"Error: distributions must have the same input space: {dist_pred.action_dims} != {dist_true.action_dims}"
        return th.stack(
            [th.distributions.kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return th.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)
