from dataclasses import dataclass
from typing import Generic, TypeVar
import gymnasium
from gymnax import EnvParams, EnvState
from gymnax.environments.spaces import Space as GymnaxSpace
import gymnax
from omegaconf import DictConfig
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import (
    Discrete as GymnaxDiscrete,
    Box as GymnaxBox,
    Tuple as GymnaxTuple,
    Dict as GymnaxDict,
)
import gymnasium as gym
from jax import numpy as jnp

from src.env_utils.jax_wrappers import (
    BatchEnv,
    BraxGymnaxWrapper,
    ClipAction,
    FlattenObsWrapper,
    LogWrapper,
    MjxGymnaxWrapper,
)
from src.env_utils.torch_wrappers.maniskill_wrapper import ManiSkillWrapper

Env = gymnasium.Env | Environment[EnvState, EnvParams]
Space = gymnasium.Space | GymnaxSpace

E = TypeVar("E", bound=Env)
S = TypeVar("S", bound=Space)


@dataclass
class EnvSetup(Generic[E]):
    env: E
    eval_env: E
    action_space: GymnaxSpace
    observation_space: GymnaxSpace


def _gymnasium_to_gymnax_space(space: gymnasium.Space) -> GymnaxSpace:
    if isinstance(space, gymnasium.spaces.Discrete):
        return GymnaxDiscrete(num_categories=space.n)
    elif isinstance(space, gymnasium.spaces.Box):
        return GymnaxBox(
            low=jnp.array(space.low),
            high=jnp.array(space.high),
            dtype=space.dtype,
            shape=space.shape,
        )
    elif isinstance(space, gymnasium.spaces.Tuple):
        return GymnaxTuple(tuple(_gymnasium_to_gymnax_space(s) for s in space.spaces))
    elif isinstance(space, gymnasium.spaces.Dict):
        return GymnaxDict(
            {k: _gymnasium_to_gymnax_space(s) for k, s in space.spaces.items()}
        )
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def _make_brax_env(cfg: DictConfig) -> EnvSetup[Environment]:
    env = BraxGymnaxWrapper(cfg.env.name)  # , episode_length=cfg.env.max_episode_steps
    env = ClipAction(env)
    env = LogWrapper(env, num_envs=cfg.algorithm.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(),
        observation_space=env.observation_space(),
    )


def _make_mjx_env(cfg: DictConfig) -> EnvSetup[Environment]:
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        asymmetric_observation=cfg.env.asymmetric_observation,
    )
    env = ClipAction(env)
    env = LogWrapper(env, num_envs=cfg.algorithm.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_gymnax_env(cfg: DictConfig) -> EnvSetup[Environment]:
    env, env_params = gymnax.make(cfg.env.name)
    env = FlattenObsWrapper(env)
    env = BatchEnv(env)
    env = LogWrapper(env, num_envs=cfg.algorithm.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_minatar_env(cfg: DictConfig) -> EnvSetup[Environment]:
    env, env_params = gymnax.make(cfg.env.name)
    env = BatchEnv(env)
    env = LogWrapper(env, num_envs=cfg.algorithm.num_envs)
    eval_env = env
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=env.action_space(env.default_params),
        observation_space=env.observation_space(env.default_params),
    )


def _make_gymnasium_env(cfg: DictConfig) -> EnvSetup[gymnasium.Env]:
    def _make():
        env = gym.make(cfg.env.name)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    env = gym.vector.SyncVectorEnv([_make for _ in range(cfg.algorithm.num_envs)])
    eval_env = gym.vector.SyncVectorEnv([_make for _ in range(cfg.algorithm.num_envs)])
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=_gymnasium_to_gymnax_space(env.single_action_space),
        observation_space=_gymnasium_to_gymnax_space(env.single_observation_space),
    )


def _make_maniskill_env(cfg: DictConfig) -> EnvSetup[gymnasium.Env]:
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    def make_env(eval: bool = False):
        env_kwargs = cfg.env.kwargs if "kwargs" in cfg.env else {}
        if cfg.env.control_mode is not None:
            env_kwargs["control_mode"] = cfg.env.control_mode
        reconfiguration_freq = (
            cfg.env.eval_reconfiguration_freq if eval else cfg.env.reconfiguration_freq
        )
        partial_resets = cfg.env.eval_partial_reset if eval else cfg.env.partial_reset
        envs = gym.make(
            cfg.env.name,
            num_envs=cfg.algorithm.num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **env_kwargs,
        )

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
        if cfg.env.capture_video:
            if cfg.env.save_train_video_freq is not None or eval:
                video_dir = "train_videos" if not eval else "eval_videos"
                save_video_trigger = (
                    lambda x: (x // cfg.algorithm.num_steps)
                    % cfg.env.save_train_video_freq
                    == 0
                )
                envs = RecordEpisode(
                    envs,
                    output_dir=video_dir,
                    save_trajectory=False,
                    save_video_trigger=save_video_trigger,
                    max_steps_per_video=cfg.algorithm.num_steps,
                    video_fps=30,
                )
        envs = ManiSkillVectorEnv(
            envs,
            cfg.algorithm.num_envs,
            ignore_terminations=not partial_resets,
            record_metrics=True,
        )
        envs = ManiSkillWrapper(
            envs,
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=partial_resets,
        )
        return envs

    env = make_env(eval=False)
    eval_env = make_env(eval=True)
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=_gymnasium_to_gymnax_space(env.single_action_space),
        observation_space=_gymnasium_to_gymnax_space(env.single_observation_space),
    )


def _make_atari_env(cfg: DictConfig) -> EnvSetup[gymnasium.Env]:
    import envpool
    from src.env_utils.atari import RecordEpisodeStatistics

    def make():
        env = envpool.make(
            cfg.env.name,
            env_type="gymnasium",
            num_envs=cfg.algorithm.num_envs,
            episodic_life=True,
            reward_clip=True,
        )
        env = RecordEpisodeStatistics(env)
        return env

    env = make()
    eval_env = make()
    return EnvSetup(
        env=env,
        eval_env=eval_env,
        action_space=_gymnasium_to_gymnax_space(env.action_space),
        observation_space=_gymnasium_to_gymnax_space(env.observation_space),
    )


def make_env(cfg: DictConfig) -> EnvSetup[Env]:
    if cfg.env.type == "brax":
        return _make_brax_env(cfg)
    elif cfg.env.type == "mjx":
        return _make_mjx_env(cfg)
    elif cfg.env.type == "gymnax":
        return _make_gymnax_env(cfg)
    elif cfg.env.type == "minatar":
        return _make_minatar_env(cfg)
    elif cfg.env.type == "gymnasium":
        return _make_gymnasium_env(cfg)
    elif cfg.env.type == "atari":
        return _make_atari_env(cfg)
    elif cfg.env.type == "maniskill":
        return _make_maniskill_env(cfg)
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")
