import flax
from flax import nnx
import jax
import jax.numpy as jnp
from gymnax.environments.spaces import Space, Box, Discrete


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs, in_features, out_features, use_norm=True, activation=nnx.swish
):
    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )
    ]
    if use_norm:
        layers.append(nnx.RMSNorm(out_features, rngs=rngs))
    if activation is not None:
        layers.append(activation)
    return nnx.Sequential(*layers)


class Identity(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm: bool = True,
        use_output_norm: bool = False,
        layers: int = 2,
        input_activation: bool = False,
        input_skip: bool = False,
        hidden_skip: bool = False,
        output_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = layers
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        if output_activation is None:
            self.output_activation = Identity()
        elif output_activation == True:
            self.output_activation = hidden_activation
        else:
            self.output_activation = output_activation

        self.input_skip = input_skip
        self.hidden_skip = hidden_skip
        self.output_skip = output_skip
        if layers == 1:
            hidden_dim = out_features
        self.input_layer = normed_activation_layer(
            rngs,
            in_features,
            hidden_dim,
            use_norm=use_norm,
            activation=hidden_activation,
        )
        self.main_layers = [
            normed_activation_layer(
                rngs,
                hidden_dim,
                hidden_dim,
                use_norm=use_norm,
                activation=hidden_activation,
            )
            for _ in range(layers - 2)
        ]
        self.norm = nnx.RMSNorm(in_features, rngs=rngs)
        self.output_layer = normed_activation_layer(
            rngs,
            hidden_dim,
            out_features,
            use_norm=use_output_norm,
            activation=self.output_activation,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        def _potentially_skip(skip, x, layer):
            if skip:
                return x + layer(x)
            else:
                return layer(x)

        if self.input_activation:
            # x = self.norm(x)
            x = self.hidden_activation(x)
        if self.layers == 1:
            return _potentially_skip(self.input_skip, x, self.input_layer)
        x = _potentially_skip(self.input_skip, x, self.input_layer)
        for layer in self.main_layers:
            x = _potentially_skip(self.hidden_skip, x, layer)
        return _potentially_skip(self.output_skip, x, self.output_layer)


class MLPStateEncoder(nnx.Module):
    def __init__(
        self,
        observation_space: Space,
        output_dim: int,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()
        if isinstance(observation_space, Box):
            self.encoder = MLP(
                in_features=observation_space.shape[0],
                out_features=output_dim,
                **kwargs,
                rngs=rngs,
            )
        elif isinstance(observation_space, Discrete):
            self.encoder = nnx.Embed(
                num_embeddings=observation_space.n,
                features=output_dim,
                rngs=rngs,
            )
        else:
            raise NotImplementedError(
                f"StateEncoder not implemented for observation space type {type(observation_space)}"
            )

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.encoder(obs)


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
        self.state_encoder = MLP(
            in_features=observation_space.shape[0],
            out_features=output_dim,
            **kwargs,
            rngs=rngs,
        )
        if isinstance(action_space, Box):
            self.action_encoder = MLP(
                in_features=action_space.shape[0],
                out_features=output_dim,
                **kwargs,
                rngs=rngs,
            )
        elif isinstance(action_space, Discrete):
            self.action_encoder = nnx.Embed(
                num_embeddings=action_space.n,
                features=output_dim,
                rngs=rngs,
            )
        else:
            raise NotImplementedError(
                f"ActionEncoder not implemented for action space type {type(action_space)}"
            )
        self.project = nnx.Linear(
            in_features=2 * output_dim,
            out_features=output_dim,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        state_embedding = self.state_encoder(obs)
        action_embedding = self.action_encoder(action)
        combined = jnp.concatenate([state_embedding, action_embedding], axis=-1)
        return self.project(combined)


class AtariCNNEncoder(nnx.Module):
    def __init__(self, output_dim: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.cnn = nnx.Sequential(
            [
                nnx.Conv(
                    in_features=4,
                    out_features=32,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    padding="VALID",
                    use_bias=False,
                    rngs=rngs,
                ),
                nnx.relu,
                nnx.Conv(
                    in_features=32,
                    out_features=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="VALID",
                    use_bias=False,
                    rngs=rngs,
                ),
                nnx.relu,
                nnx.Conv(
                    in_features=64,
                    out_features=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="VALID",
                    use_bias=False,
                    rngs=rngs,
                ),
                nnx.relu,
            ]
        )
        self.project = nnx.Linear(
            in_features=7 * 7 * 64,
            out_features=output_dim,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs / 255.0
        x = self.cnn(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = self.project(x)
        return x
