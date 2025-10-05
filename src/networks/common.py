import flax
from flax import nnx
import jax
import jax.numpy as jnp
from gymnax.environments.spaces import Space, Box, Discrete


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs, in_features, out_features, use_norm=True, activation=nnx.swish, scale=1.0
):
    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            kernel_init=nnx.initializers.orthogonal(scale=scale),
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
        final_layer_scaling: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = layers
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        if output_activation is None:
            self.output_activation = Identity()
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
        self.main_layers = nnx.List([
            normed_activation_layer(
                rngs,
                hidden_dim,
                hidden_dim,
                use_norm=use_norm,
                activation=hidden_activation,
            )
            for _ in range(layers - 2)
        ])
        self.norm = nnx.LayerNorm(in_features, rngs=rngs)
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


class StateActionInput(nnx.Module):
    def __init__(
        self,
        state_encoder: nnx.Module | None = Identity(),
        action_encoder: nnx.Module | None = Identity(),
        concatenate: bool = True,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.concatenate = concatenate

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        if self.state_encoder is not None:
            obs = self.state_encoder(obs)
        else:
            obs = None
        if self.action_encoder is not None:
            action = self.action_encoder(action)
        else: 
            action = None

        if obs is None and action is None:
            raise ValueError("Both state_encoder and action_encoder cannot be None.")
        elif action is None:
            return obs
        elif obs is None:
            return action
    
        if not self.concatenate:
            return obs, action
        else:
            return jnp.concatenate([obs, action], axis=-1)
