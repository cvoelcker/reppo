import distrax
import jax
import jax.numpy as jnp
from flax import nnx
import math


class DiscretePolicyHead(nnx.Module):
    def __call__(
        self,
        features: jax.Array,
        deterministic: bool = False,
        scale: None | jax.Array = None,
    ) -> distrax.Categorical | jax.Array:
        if scale is None:
            scale = 1.0
        logits = jax.nn.log_softmax(features, axis=-1)
        if deterministic:
            return jnp.argmax(logits, axis=-1)
        else:
            dist = distrax.Categorical(logits=logits / scale)
            return dist


class TanhGaussianPolicyHead(nnx.Module):
    def __init__(
        self,
        min_std: float = 1e-1,
        fixed_std: bool = False,
    ):
        self.min_std = min_std
        self.fixed_std = fixed_std
        self.std_param = nnx.Param(jnp.ones(1) * math.log(0.6)) if fixed_std else None

    def __call__(
        self,
        features: jax.Array,
        deterministic: bool = False,
        scale: None | jax.Array = None,
    ) -> distrax.Distribution | jax.Array:
        if scale is None:
            scale = 1.0
        mean, log_std = jnp.split(features, 2, axis=-1)
        if self.fixed_std:
            log_std = jnp.ones_like(log_std) * self.std_param
        if deterministic:
            return jnp.tanh(mean)
        else:
            std = (jnp.exp(log_std) + self.min_std) * scale
            pi = distrax.Transformed(
                distrax.Normal(loc=mean, scale=std), distrax.Tanh()
            )
            pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
            return pi
