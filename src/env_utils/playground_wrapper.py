from functools import partial
import functools
from typing import Any, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import wrap_for_brax_training, Wrapper
import numpy as np


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        if isinstance(env_or_name, str):
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
        else:
            self.env = env_or_name
        self.reward_scale = reward_scale
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True
        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        self.asymmetric_observation = asymmetric_observation
        self.episode_length = episode_length
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        if self.asymmetric_observation:
            return gymnax.environments.spaces.Dict(
                {
                    "state": gymnax.environments.spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=self.env.observation_size["state"],
                    ),
                    "privileged_state": gymnax.environments.spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=self.env.observation_size["privileged_state"],
                    ),
                }
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def _get_obs(self, state):
        if self.asymmetric_observation:
            obs = {
                "state": state.obs["state"] if self.dict_obs else state.obs[..., 0, :],
                "privileged_state": state.obs["privileged_state"]
                if self.dict_obs
                else state.obs[..., 1, :],
            }
        else:
            obs = state.obs
        return obs

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = self._get_obs(state)
        return obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        state = self.env.step(state, action)
        obs = self._get_obs(state)
        return (
            obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            state.info.copy(),
        )
