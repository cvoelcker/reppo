from dataclasses import dataclass
from typing import Generic, TypeVar
import gymnasium
from gymnax import EnvParams, EnvState
from gymnax.environments.spaces import Space as GymnaxSpace
import gymnax
from mujoco_playground import MjxEnv
from omegaconf import DictConfig
from gymnax.environments.environment import Environment
from brax.envs.base import State as BraxState
import gymnasium as gym

from src.env_utils.jax_wrappers import (
    BatchEnv,
    BraxGymnaxWrapper,
    ClipAction,
    FlattenObsWrapper,
    LogWrapper,
    MjxGymnaxWrapper,
)

Env = gymnasium.Env | Environment[EnvState, EnvParams]
Space = gymnasium.Space | GymnaxSpace

E = TypeVar("E", bound=Env)
S = TypeVar("S", bound=Space)


@dataclass
class EnvSetup(Generic[E, S]):
    env: E
    eval_env: E
    action_space: S
    observation_space: S


def _make_brax_env(cfg: DictConfig) -> EnvSetup[Environment, GymnaxSpace]:
    env = BraxGymnaxWrapper(cfg.env.name)  # , episode_length=cfg.env.max_episode_steps
    env = ClipAction(env)
    env = LogWrapper(env, num_envs=cfg.hyperparameters.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_mjx_env(cfg: DictConfig) -> EnvSetup[Environment, GymnaxSpace]:
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        asymmetric_observation=cfg.env.asymmetric_observation,
    )
    env = ClipAction(env)
    env = LogWrapper(env, num_envs=cfg.hyperparameters.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_gymnax_env(cfg: DictConfig) -> EnvSetup[Environment, GymnaxSpace]:
    env, env_params = gymnax.make(cfg.env.name)
    env = FlattenObsWrapper(env)
    env = BatchEnv(env)
    env = LogWrapper(env, num_envs=cfg.hyperparameters.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_gymnasium_env(cfg: DictConfig) -> EnvSetup[gymnasium.Env, gymnasium.Space]:
    def _make():
        env = gym.make(cfg.env.name)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    env = gym.vector.AsyncVectorEnv([_make for _ in range(cfg.hyperparameters.num_envs)])
    eval_env = gym.vector.AsyncVectorEnv(
        [_make for _ in range(cfg.hyperparameters.num_envs)]
    )
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.single_action_space,
        observation_space=env.single_observation_space,
    )


def make_env(cfg: DictConfig) -> EnvSetup[Env, Space]:
    if cfg.env.type == "brax":
        return _make_brax_env(cfg)
    elif cfg.env.type == "mjx":
        return _make_mjx_env(cfg)
    elif cfg.env.type == "gymnax":
        return _make_gymnax_env(cfg)
    elif cfg.env.type == "gymnasium":
        return _make_gymnasium_env(cfg)
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")
