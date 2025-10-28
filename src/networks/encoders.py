from flax import nnx
import jax
import jax.numpy as jnp
from gymnax.environments.spaces import Space, Box, Discrete

from src.networks.common import MLP


class MLPStateActionEncoder(nnx.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        output_dim: int,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()

        if isinstance(action_space, Discrete):
            self.action_embed = nnx.Embed(
                num_embeddings=action_space.n,
                features=output_dim,
                rngs=rngs,
            )
            action_dim = output_dim
        elif isinstance(action_space, Box):
            action_dim = action_space.shape[0]
            self.action_embed = None

        self.encoder = MLP(
            in_features=observation_space.shape[0] + action_dim,
            out_features=output_dim,
            **kwargs,
            rngs=rngs,
        )
        self.use_embed = isinstance(action_space, Discrete)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        if self.use_embed:
            action = self.action_embed(action)
        combined = jnp.concatenate([obs, action], axis=-1)
        return self.encoder(combined)


class AtariCNNEncoder(nnx.Module):
    def __init__(self, output_dim: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.cnn = nnx.Sequential(
            nnx.Conv(
                in_features=4,
                out_features=32,
                kernel_size=(8, 8),
                strides=(4, 4),
                kernel_init=nnx.initializers.he_normal(),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=32,
                out_features=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                kernel_init=nnx.initializers.he_normal(),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=64,
                out_features=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_init=nnx.initializers.he_normal(),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.relu,
        )
        self.project = nnx.Linear(
            in_features=7 * 7 * 64,
            out_features=output_dim,
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32) / 255.0
        x = jnp.swapaxes(x, -1, -3)  # CHW to HWC
        x = self.cnn(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = self.project(x)
        x = nnx.relu(x)
        return x


class MinatarConvNet(nnx.Module):
    def __init__(
        self,
        in_features: int = 4,
        normalization: nnx.Module = nnx.LayerNorm,
        hidden_dim: int = 512,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv_layers = nnx.Sequential(
            nnx.Conv(
                in_features=in_features,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                rngs=rngs,
            ),
            normalization(16, rngs=rngs),
            nnx.relu,
        )
        self.fc_layers = nnx.Sequential(
            nnx.Linear(
                in_features=1024,
                out_features=hidden_dim,
                rngs=rngs,
            ),
            normalization(hidden_dim, rngs=rngs),
            nnx.relu,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_layers(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = self.fc_layers(x)
        return x
