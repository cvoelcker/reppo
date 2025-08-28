

import distrax
import gymnasium
import gymnax
import jax
from flax import nnx
from jax import numpy as jnp




class PPONetworks(nnx.Module):
    def __init__(
        self,
        obs_space: gymnax.environments.spaces.Space,
        action_space: gymnax.environments.spaces.Space,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.discrete_action = isinstance(
            action_space, gymnax.environments.spaces.Discrete | gymnasium.spaces.Discrete)
        print("Discrete action space:", self.discrete_action)
        if (
            isinstance(obs_space, gymnax.environments.spaces.Dict)
            and "privileged_state" in obs_space.spaces
        ):
            self.asymmetric_obs = True
            critic_obs_dim = obs_space.spaces["privileged_state"].shape[-1]
            obs_dim = obs_space.spaces["state"].shape[-1]
        else:
            self.asymmetric_obs = False
            critic_obs_dim = obs_space.shape[-1]
            obs_dim = obs_space.shape[-1]

        if self.discrete_action:
            action_dim = action_space.n
        else:
            action_dim = action_space.shape[-1]
            self.log_std = nnx.Param(jnp.zeros(action_dim))

        def linear_layer(in_features, out_features, scale=jnp.sqrt(2)):
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                kernel_init=nnx.initializers.orthogonal(scale=scale),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        self.actor_module = nnx.Sequential(
            linear_layer(obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, action_dim, scale=0.01),
        )

        self.critic_module = nnx.Sequential(
            linear_layer(critic_obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, 1, scale=1.0),
        )

    def critic(self, obs: jax.Array) -> jax.Array:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "privileged_state" in obs
            ), "Privileged state must be provided for asymmetric observations."
            obs = obs["privileged_state"]
        return self.critic_module(obs).squeeze()

    def actor(self, obs: jax.Array) -> distrax.Distribution:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]
        loc = self.actor_module(obs)
        if self.discrete_action:
            pi = distrax.Categorical(logits=loc)
        else:
            pi = distrax.MultivariateNormalDiag(
                loc=loc, scale_diag=jnp.exp(self.log_std.value)
            )
        return pi