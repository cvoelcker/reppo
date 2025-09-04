import math
from typing import Sequence, Union

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.algorithms import utils
from src.networks import (
    MLP,
    ContinuousCategoricalCriticHead,
    DiscreteCategoricalCriticHead,
)


class ContinuousCategoricalQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        encoder_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = MLP(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.q_head = ContinuousCategoricalCriticHead(
            in_features=hidden_dim,
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> dict[str, jax.Array]:
        inputs = jnp.concatenate([obs, action], axis=-1)
        features = self.encoder(inputs)
        q_inputs = nnx.swish(features)
        q_output = self.q_head(q_inputs)
        q_output["embed"] = features
        return q_output


class DiscreteCategoricalQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        encoder_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = MLP(
            in_features=obs_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.q_head = DiscreteCategoricalCriticHead(
            in_features=hidden_dim,
            num_actions=num_actions,
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array | None = None) -> dict[str, jax.Array]:
        features = self.encoder(obs)
        q_inputs = nnx.swish(features)
        q_output = self.q_head(q_inputs, action)
        q_output["embed"] = features
        return q_output


class ContinuousQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        encoder_layers: int = 1,
        head_layers: int = 1,
        use_skip=False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.encoder = MLP(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.q_head = MLP(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            hidden_skip=use_skip,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> dict[str, jax.Array]:
        inputs = jnp.concatenate([obs, action], axis=-1)
        features = self.encoder(inputs)
        q_inputs = nnx.swish(features)
        q_value = self.q_head(q_inputs)
        return {"value": q_value, "embed": features}


class DiscreteQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_encoder_norm: bool = False,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        use_skip=False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.encoder = MLP(
            in_features=obs_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.q_head = MLP(
            in_features=hidden_dim,
            out_features=num_actions,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            hidden_skip=use_skip,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array = None) -> dict[str, jax.Array]:
        features = self.encoder(obs)
        q_inputs = nnx.swish(features)
        values = self.q_head(q_inputs)
        if action is not None:
            values = jnp.take_along_axis(values, action, axis=-1)
        return {"value": values, "embed": features}


class ContinuousActorNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.policy = MLP(
            in_features=obs_dim,
            out_features=action_dim * 2,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def __call__(
        self,
        obs: jax.Array,
        deterministic: bool = False,
        scale: float | jax.Array = 1.0,
    ) -> distrax.Distribution | jax.Array:
        loc = self.policy(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        if deterministic:
            action = jnp.tanh(loc)
            return action
        else:
            std = (jnp.exp(log_std) + self.min_std) * scale
            pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
            pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
            return pi

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)


class DiscreteActorNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.policy = MLP(
            in_features=obs_dim,
            out_features=num_actions,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def __call__(
        self,
        obs: jax.Array,
        deterministic: bool = False,
        scale: float | jax.Array = 1.0,
    ) -> distrax.Distribution | jax.Array:
        logits = self.policy(obs)
        if deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action
        else:
            pi = distrax.Categorical(logits=logits / scale)
        return pi

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)
