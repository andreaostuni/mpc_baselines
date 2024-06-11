import os

from setuptools import find_packages, setup

with open(os.path.join("mpc_baselines", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

long_description = """

# MPC Baselines

This repository contains the implementation of Model Predictive Control (MPC) baselines for Reinforcement Learning (RL) in PyTorch.

## Links

Repository:
https://github.com/andreaostuni/mpc-baselines

## Quick example

This library is designed to be used with [Stable Baselines3](https://stable-baselines3.readthedocs.io/),
a set of reliable implementations of reinforcement learning algorithms in PyTorch.

It implements the MPC policies for the following algorithms:
- [PPO] (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

```python
import gymnasium

from stable_baselines3 import PPO

env = gymnasium.make("CartPole-v1", render_mode="human")

model = PPO("MPCMlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

```


"""  
setup(
    name="mpc-baselines",
    packages=[package for package in find_packages() if package.startswith("mpc_baselines")],
    package_data={"mpc_baselines": ["py.typed", "version.txt"]},
    install_requires=[
        "gymnasium>=0.28.1,<0.30",
        "numpy>=1.20",
        "torch>=1.13",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
        # for integration with SB3
        "stable-baselines3>=1.0.0",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.3.1",
            # Reformat
            "black>=24.2.0,<25",
        ],
        "docs": [
            "sphinx>=5,<8",
            "sphinx-autobuild",
            "sphinx-rtd-theme>=1.3.0",
            # For spelling
            "sphinxcontrib.spelling",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
    },
    description="Pytorch version of MPC Baselines",
    author="Andrea Ostuni",
    url="https://github.com/andreaostuni/mpc-baselines",
    author_email="andrea.ostuni@polito.it",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gymnasium gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.8",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/andreaostuni/mpc-baselines",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
