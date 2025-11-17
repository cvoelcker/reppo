
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from gymnax.environments import spaces


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
        state = jax.vmap(self.env.reset)(key)
        return state.obs, state

    def step(self, key, state, action):
        next_state = jax.vmap(self.env.step)(state, action)
        return (
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
        )

    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )
