import jax
import jax.numpy as jnp
from gymnax.environments.spaces import Space, Box, Discrete
from flax import nnx

from src.algorithms import utils
from src.networks.common import MLP


class CategoricalContinuousQNetworkHead(nnx.Module):
    def __init__(
        self,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def __call__(self, features: jax.Array) -> jax.Array:
        logits = features + self.zero_dist.value * 40.0
        probs = jax.nn.softmax(logits, axis=-1)
        value = probs.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return {
            "value": value,
            "probs": probs,
            "logits": logits,
        }


class CategoricalDiscreteQNetworkHead(nnx.Module):
    def __init__(
        self,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def __call__(self, features: jax.Array, action: jax.Array = None) -> jax.Array:
        features = features.reshape(*features.shape[:-1], -1, self.num_bins)
        logits = features + self.zero_dist.value * 40.0
        if action is not None:
            logits = jnp.take_along_axis(
                logits,
                action[..., None, None],
                axis=-2,
            ).squeeze(-2)
        probs = jax.nn.softmax(logits, axis=-1)
        value = probs.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return {
            "value": value,
            "probs": probs,
            "logits": logits,
        }
