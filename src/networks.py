import math
from typing import Sequence, Union

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.algorithms import utils


def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
):
    "TODO: push to jax"
    return nnx.initializers.variance_scaling(
        0.3333,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


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
            activation=output_activation,
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


class ContinuousCategoricalCriticHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        *,
        rngs: nnx.Rngs,
    ):
        self._num_bins = num_bins
        self._vmin = vmin
        self._vmax = vmax
        self._value_bins = jnp.linspace(vmin, vmax, num_bins, endpoint=True)
        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )
        self.linear = nnx.Linear(
            in_features=in_features, out_features=num_bins, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        logits = self.linear(x) + self.zero_dist.value * 40.0
        probs = jax.nn.softmax(logits, axis=-1)
        value = probs.dot(self._value_bins)
        return {"logits": logits, "probs": probs, "value": value}


class DiscreteCategoricalCriticHead(nnx.Module):

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        *,
        rngs: nnx.Rngs,
    ):
        self._num_bins = num_bins
        self._vmin = vmin
        self._vmax = vmax
        self._num_actions = num_actions
        self._value_bins = jnp.linspace(vmin, vmax, num_bins, endpoint=True)
        self._zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((num_actions,)), num_bins, vmin, vmax)
        )
        self.linear = nnx.Linear(
            in_features=in_features, out_features=num_actions * num_bins, rngs=rngs
        )

    def __call__(self, embed: jax.Array, action: int = None) -> dict[str, jax.Array]:
        proj = self.linear(embed)
        logits_per_action = proj.reshape(-1, self._num_actions, self._num_bins) + self._zero_dist.value * 40.0
        if action is not None:
            logits = jnp.take_along_axis(logits_per_action, action, axis=-2)
        else:
            logits = logits_per_action
        probs = jax.nn.softmax(logits, axis=-1)
        value = probs.dot(self._value_bins)
        return {"logits": logits, "probs": probs, "value": value}