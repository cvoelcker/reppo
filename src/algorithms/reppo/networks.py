import math
from typing import Sequence, Union

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.algorithms import utils
from gymnax.environments.spaces import Box, Discrete


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


class FCNN(nnx.Module):
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


class CriticNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
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
        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            input_activation=not use_simplical_embedding,
            hidden_skip=use_skip,
            layers=head_layers,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            input_activation=not use_simplical_embedding,
            layers=pred_layers,
            rngs=rngs,
        )

    def features(self, obs: jax.Array, action: jax.Array):
        state = jnp.concatenate([obs, action], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def forward(self, obs, action):
        features = self.features(obs, action)
        value = self.critic_head(features)
        pred = self.pred_module(features)
        pred_rew = pred[..., :1]
        pred_features = pred[..., 1:]
        return features, pred_features, pred_rew, value.squeeze(-1)


class ContinuousCategoricalCriticNetwork(nnx.Module):
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.use_skip = use_skip

        if project_discrete_action:
            self.action_embedding = nnx.Embed(
                num_embeddings=action_space.shape[0],
                features=hidden_dim // 2,
            )
            action_dim = hidden_dim // 2
        else:
            self.action_embedding = Identity()

        self.feature_module = FCNN(
            in_features=observation_space.shape[0] + action_space.shape[0],
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
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=self.num_bins,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=None,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            rngs=rngs,
        )

        self.zero_dist = nnx.Variable(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def features(self, obs: jax.Array, action: jax.Array):
        action_embedding = self.action_embedding(action)
        state = jnp.concatenate([obs, action_embedding], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features) + self.zero_dist.value * 40.0
        return cat

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat(obs, action), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def __call__(self, obs, action):
        features = self.features(obs, action)
        logits = self.critic_head(features)
        value_cat = jax.nn.softmax(logits, axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        preds = self.pred_module(features)
        pred_rew = preds[..., :1]
        pred_features = preds[..., 1:]
        if self.use_skip:
            pred_features = pred_features + features

        return {
            "embed": pred_features,
            "pred_rew": pred_rew,
            "value": value,
            "probs": value_cat,
            "logits": logits,
        }


class DiscreteCategoricalCriticNetwork(nnx.Module):
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.num_actions = action_space.n
        self.use_skip = use_skip

        self.feature_module = FCNN(
            in_features=observation_space.shape[0],
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
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=self.num_bins * self.num_actions,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=None,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            rngs=rngs,
        )

        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def features(self, obs: jax.Array):
        return self.feature_module(obs)

    def critic_head(self, features: jax.Array) -> jax.Array:
        out = self.critic_module(features)
        cat = (
            out.reshape(*features.shape[:-1], -1, self.num_bins)
            + self.zero_dist.value * 40.0
        )
        return cat

    def critic_cat(self, obs: jax.Array) -> jax.Array:
        features = self.features(obs)
        return self.critic_head(features)

    def critic(self, obs: jax.Array, action: jax.Array = None) -> jax.Array:
        features = self.features(obs)
        logits = self.critic_head(features)
        if action is not None:
            logits = logits.reshape(-1, self.num_actions, self.num_bins)
            logits = jnp.take_along_axis(
                logits, action.reshape(-1, 1, 1), axis=1
            ).squeeze(1)
            logits = logits.reshape(*features.shape[:-1], -1)
        value_cat = jax.nn.softmax(logits, axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def __call__(self, obs, action=None):
        features = self.features(obs)
        logits = self.critic_head(features)
        if action is not None:
            logits = logits.reshape(-1, self.num_actions, self.num_bins)
            logits = jnp.take_along_axis(
                logits, action.reshape(-1, 1, 1), axis=1
            ).squeeze(1)
            logits = logits.reshape(*features.shape[:-1], -1)
        value_cat = jax.nn.softmax(logits, axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        preds = self.pred_module(features)
        pred_rew = preds[..., :1]
        pred_features = preds[..., 1:]
        if self.use_skip:
            pred_features = pred_features + features

        return {
            "embed": pred_features,
            "pred_rew": pred_rew,
            "value": value,
            "probs": value_cat,
            "logits": logits,
        }


class SACActorNetworks(nnx.Module):
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
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
        self.actor_module = FCNN(
            in_features=observation_space.shape[0],
            out_features=action_space.shape[0] * 2,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std
        self.d = action_space.shape[0]

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    def __call__(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, std = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


class GumbleSoftmaxDistribution(distrax.Distribution):
    def __init__(self, logits: jax.Array, temperature: jax.Array):
        self.logits = logits
        self.temperature = temperature

    def sample(self, seed=None):
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).log_prob(value)

    def sample_and_log_prob(self, *, seed, sample_shape=...):
        sample = self.sample(seed=seed)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class SACDiscreteActorNetworks(nnx.Module):
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
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
        self.actor_module = FCNN(
            in_features=observation_space.shape[0],
            out_features=action_space.n,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std
        self.d = action_space.n

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        pi = distrax.Categorical(logits=loc / scale)
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        return jnp.argmax(loc, axis=-1)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)



class Critic(nnx.Module):
    def __init__(
        self,
        q_network: ContinuousCategoricalCriticNetwork | DiscreteCategoricalCriticNetwork,
        asymmetric_obs: bool = False,
    ):
        self.q_network = q_network
        self.asymmetric_obs = asymmetric_obs

    def __call__(self, obs: jax.Array, action: jax.Array = None) -> jax.Array:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "privileged_state" in obs
            ), "Privileged state must be provided for asymmetric observations."
            obs = obs["privileged_state"]
        return self.q_network(obs, action)
    
class Actor(nnx.Module):
    def __init__(
        self,
        actor_network: SACActorNetworks | SACDiscreteActorNetworks,
        asymmetric_obs: bool = False,
    ):
        self.actor_network = actor_network
        self.asymmetric_obs = asymmetric_obs
        self.d = actor_network.d

    def __call__(self, obs: jax.Array, scale: jax.Array = 1.0) -> distrax.Distribution:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]
        return self.actor_network.actor(obs, scale=scale)
    
    def det_action(self, obs: jax.Array) -> jax.Array:
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]
        return self.actor_network.det_action(obs)
    
    def temperature(self) -> jax.Array:
        return self.actor_network.temperature()
    
    def lagrangian(self) -> jax.Array:
        return self.actor_network.lagrangian()