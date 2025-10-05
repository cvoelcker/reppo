import distrax
import gymnasium
import gymnax
import jax
from flax import nnx
from jax import numpy as jnp
from gymnax.environments.spaces import Discrete
from src.networks.encoders import AtariCNNEncoder


class PPONetworks(nnx.Module):
    def __init__(
        self,
        obs_space: gymnax.environments.spaces.Space,
        action_space: gymnax.environments.spaces.Space,
        hidden_dim: int = 64,
        use_tanh_gaussian: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.discrete_action = isinstance(
            action_space,
            gymnax.environments.spaces.Discrete | gymnasium.spaces.Discrete,
        )
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
        self.use_tanh_gaussian = use_tanh_gaussian

    def critic(self, obs: jax.Array) -> jax.Array:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "privileged_state" in obs
            ), "Privileged state must be provided for asymmetric observations."
            obs = obs["privileged_state"]
        return self.critic_module(obs).squeeze()

    def actor(self, obs: jax.Array, deterministic: bool = False, mean_offset=0.0) -> distrax.Distribution:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]
        loc = self.actor_module(obs) + mean_offset
        if self.discrete_action:
            pi = distrax.Categorical(logits=loc)
        else:
            if self.use_tanh_gaussian:
                if deterministic:
                    return jnp.tanh(loc)
                std = jnp.exp(self.log_std.value)
                # pi = distrax.MultivariateNormalDiag(
                #     loc=loc, scale_diag=loc
                # )
                pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
                pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
            else:
                if deterministic:
                    return loc
                std = jnp.exp(self.log_std.value)
                pi = distrax.MultivariateNormalDiag(loc=loc, scale_diag=std)
        return pi
    

class PPOAtariNetworks(nnx.Module):
    def __init__(
        self,
        obs_space: gymnax.environments.spaces.Space,
        action_space: Discrete,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        def linear_layer(in_features, out_features, scale=jnp.sqrt(2)):
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                kernel_init=nnx.initializers.orthogonal(scale=scale),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        
        actor_cnn = AtariCNNEncoder(output_dim=hidden_dim, rngs=rngs)
        critic_cnn = AtariCNNEncoder(output_dim=hidden_dim, rngs=rngs)

        self.actor_module = nnx.Sequential(
            actor_cnn,
            nnx.relu,
            linear_layer(hidden_dim, action_space.n, scale=0.01)
        )

        self.critic_module = nnx.Sequential(
            critic_cnn,
            nnx.relu,
            linear_layer(hidden_dim, 1, scale=1.0),
        )

    def critic(self, obs: jax.Array) -> jax.Array:
        return self.critic_module(obs).squeeze()

    def actor(self, obs: jax.Array) -> distrax.Distribution:
        loc = self.actor_module(obs)
        pi = distrax.Categorical(logits=loc)
        return pi
