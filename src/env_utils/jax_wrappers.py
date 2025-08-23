from functools import partial
from typing import Any, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import wrap_for_brax_training, Wrapper


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
        print(self.dict_obs_key)
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
                "privileged_state": state.obs["privileged_state"] if self.dict_obs else state.obs[..., 1, :],
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
            {},
        )


class BatchEnv(Wrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(self, key):
        obs, env_state = jax.vmap(self.env.reset)(key)
        return obs, env_state

    def step(self, key, state, action):
        obs, env_state, reward, done, info = jax.vmap(self.env.step)(
            key, state.env_state, action
        )
        return obs, env_state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self.env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self.num_envs,)),
            episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((self.num_envs,)),
            returned_episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            timestep=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            truncated=jnp.ones((self.num_envs,), dtype=jnp.float32),
            info={
                "returned_episode": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "returned_episode_returns": jnp.zeros((self.num_envs,)),
                "timestep": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                "returned_episode_lengths": jnp.zeros(
                    (self.num_envs,), dtype=jnp.int32
                ),
            },
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        info["returned_episode_returns"] = (
            state.returned_episode_returns * (1 - done) + new_episode_return * done
        )
        info["returned_episode_lengths"] = (
            state.returned_episode_lengths * (1 - done) + new_episode_length * done
        )
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            truncated=env_state.info["truncation"],
            info=info,
        )
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(
        self,
        env_name,
        backend="generalized",
        episode_length=1000,
        reward_scaling=1.0,
        terminate=True,
    ):
        env = envs.get_environment(
            env_name=env_name, backend=backend, terminate_when_unhealthy=terminate
        )
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self.env = env
        self.action_size = self.env.action_size
        self.observation_size = (self.env.observation_size,)
        self.default_params = ()
        self.reward_scaling = reward_scaling

    def reset(self, key):
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action):
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state.obs,
            next_state,
            next_state.reward * self.reward_scaling,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        ), spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        )

    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )


class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _init_state(self, key):
        obs, env_state = self.env.reset(key)
        return NormalizeVecObsEnvState(
            mean=jax.tree.map(lambda x: jnp.mean(x, axis=0), obs),
            var=jax.tree.map(lambda x: jnp.var(x, axis=0), obs),
            count=jax.tree.map(lambda x: x.shape[0], obs),
            env_state=env_state,
        )

    def _normalize_obs(self, obs, mean, var):
        return (obs - mean) / jnp.sqrt(var + 1e-2)

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var, tot_count

    def reset(self, key, params=None):
        obs, env_state = self.env.reset(key)
        if params is not None:
            mean = params.mean
            var = params.var
            count = params.count
        else:
            mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), obs)
            var = jax.tree.map(lambda x: jnp.var(x, axis=0), obs)
            count = jax.tree.map(lambda x: x.shape[0], obs)

        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            jax.tree.map(self._normalize_obs, obs, state.mean, state.var),
            state,
        )

    def step(self, key, state, action):
        obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )

        stats = jax.tree.map(
            lambda m, v, c, o: self._compute_stats(m, v, c, o),
            state.mean,
            state.var,
            state.count,
            obs
        )

        state = NormalizeVecObsEnvState(
            mean=jax.tree.map(lambda x: x[0], stats),
            var=jax.tree.map(lambda x: x[1], stats),
            count=jax.tree.map(lambda x: x[2], stats),
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            jax.tree.map(self._normalize_obs, obs, state.mean, state.var),
            state,
            reward,
            done,
            info,
        )
