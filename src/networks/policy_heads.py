import distrax
import jax
import jax.numpy as jnp
from gymnax.environments.spaces import Space, Box, Discrete
from flax import nnx

from src.algorithms import utils
from src.networks.common import MLP


class DiscretePolicyHead(nnx.Module):


    def __call__(self, features: jax.Array, deterministic: bool = False, scale: jax.Array = 1.0) -> distrax.Categorical | jax.Array:
        logits = jax.nn.log_softmax(features, axis=-1)
        if deterministic:
            return jnp.argmax(logits, axis=-1)
        else:
            dist = distrax.Categorical(logits=logits / scale)
            return dist
    

class TanhGaussianPolicyHead(nnx.Module):
    def __init__(
        self,
        min_std: float = 1e-6,
    ):
        self.min_std = min_std

    def __call__(self, features: jax.Array, deterministic: bool = False, scale: jax.Array = 1.0) -> distrax.Distribution | jax.Array:
        mean, log_std = jnp.split(features, 2, axis=-1)
        if deterministic:
            return jnp.tanh(mean)
        else:
            std = (jnp.exp(log_std) + self.min_std) * scale
            pi = distrax.Transformed(distrax.Normal(loc=mean, scale=std), distrax.Tanh())
            pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
            return pi