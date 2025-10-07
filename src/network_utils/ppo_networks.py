import gymnax
from flax import nnx
import jax
import jax.numpy as jnp
import distrax


def linear_layer(in_features, out_features, scale=jnp.sqrt(2)):
    return nnx.Linear(
        in_features=in_features,
        out_features=out_features,
        kernel_init=nnx.initializers.orthogonal(scale=scale),
        bias_init=nnx.initializers.zeros_init(),
    )


def make_mlp(activation, input_dim, hidden_layers, output_dim):
    layers = []
    for hidden_dim in hidden_layers:
        layers.append(linear_layer(input_dim, hidden_dim))
        layers.append(activation)
        input_dim = hidden_dim
    layers.append(linear_layer(input_dim, output_dim))
    return nnx.Sequential(*layers)


class Critic(nnx.Module):
    def __init__(
        self, obs_space: gymnax.environments.spaces.Space, hidden_layers: list[int]
    ):
        super().__init__()
        self.critic_module = make_mlp(nnx.tanh, obs_space.shape[-1], hidden_layers, 1)

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.critic_module(obs).squeeze()


class ContinuousActor(nnx.Module):
    def __init__(
        self,
        obs_space: gymnax.environments.spaces.Space,
        action_space: gymnax.environments.spaces.Box,
        hidden_layers: list[int],
    ):
        super().__init__()
        self.actor_module = make_mlp(
            nnx.tanh, obs_space.shape[-1], hidden_layers, action_space.shape[-1]
        )
        self.log_std = nnx.Param(jnp.zeros(action_space.shape[-1]))

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        loc = self.actor_module(obs)
        return distrax.MultivariateNormalDiag(
            loc=loc, scale_diag=jnp.exp(self.log_std.value)
        )


class DiscreteActor(nnx.Module):
    def __init__(
        self,
        obs_space: gymnax.environments.spaces.Space,
        action_space: gymnax.environments.spaces.Discrete,
        hidden_layers: list[int],
    ):
        super().__init__()
        self.actor_module = make_mlp(
            nnx.tanh, obs_space.shape[-1], hidden_layers, action_space.n
        )

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        loc = self.actor_module(obs)
        return distrax.Categorical(logits=loc)
