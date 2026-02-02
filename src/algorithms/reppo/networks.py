import math

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from gymnax.environments.spaces import Box, Discrete

from src.networks.encoders import AtariCNNEncoder, MinatarConvNet
from src.networks.policy_heads import DiscretePolicyHead, TanhGaussianPolicyHead
from src.networks.value_heads import (
    CategoricalContinuousQNetworkHead,
    CategoricalDiscreteQNetworkHead,
)
from src.networks.common import MLP, Identity, StateActionInput


class Critic(nnx.Module):
    def __init__(
        self,
        feature_encoder: nnx.Module,
        q_network: nnx.Module,
        prediction_network: nnx.Module = None,
        asymmetric_obs: bool = False,
        is_discrete: bool = False,
    ):
        self.feature_encoder = feature_encoder
        self.q_network = q_network
        self.prediction_network = prediction_network
        self.asymmetric_obs = asymmetric_obs
        self.is_discrete = is_discrete

    def __call__(self, obs: jax.Array, action: jax.Array = None) -> jax.Array:
        if self.asymmetric_obs:
            assert isinstance(obs, dict) and "privileged_state" in obs, (
                "Privileged state must be provided for asymmetric observations."
            )
            obs = obs["privileged_state"]

        if self.is_discrete:
            features = self.feature_encoder(obs)
            output = self.q_network(features, action)
        else:
            features = self.feature_encoder(obs, action)
            output = self.q_network(features)

        output["embed"] = features
        if self.prediction_network is not None:
            pred = self.prediction_network(features)
            output["pred_features"] = pred[..., :-1]
            output["pred_rew"] = pred[..., -1:]
        return output


class Actor(nnx.Module):
    def __init__(
        self,
        feature_encoder: nnx.Module,
        policy_head: nnx.Module,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        asymmetric_obs: bool = False,
    ):
        self.feature_encoder = feature_encoder
        self.policy_head = policy_head
        self.asymmetric_obs = asymmetric_obs
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

    def __call__(self, obs: jax.Array, scale: jax.Array = 1.0) -> distrax.Distribution:
        if self.asymmetric_obs:
            assert isinstance(obs, dict) and "state" in obs, (
                "State must be provided for actor."
            )
            obs = obs["state"]
        
        # Observations should already be filtered to 25 dims by the environment wrapper
        features = self.feature_encoder(obs)
        return self.policy_head(features, scale=scale, deterministic=False)

    def det_action(self, obs: jax.Array) -> jax.Array:
        if self.asymmetric_obs:
            assert isinstance(obs, dict) and "state" in obs, (
                "State must be provided for actor."
            )
            obs = obs["state"]
        
        # Observations should already be filtered to 25 dims by the environment wrapper
        features = self.feature_encoder(obs)
        return self.policy_head(features, deterministic=True)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)


def make_continuous_actor(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Box,
    encoder: nnx.Module = None,
    *,
    rngs: nnx.Rngs,
) -> Actor:
    hparams = cfg.algorithm
    if cfg.env.get("asymmetric_observation", False):
        actor_observation_space = observation_space.spaces["state"]  # 25 # 25 for bc and - observation_space.spaces["state"] for non-bc
    else:
        actor_observation_space = observation_space  # 25 for bc and - observation_space for non-bc observation_space
    if encoder is not None:
        actor_encoder = encoder
    else:
        actor_encoder = MLP(
            in_features=actor_observation_space.shape[0], # actor_observation_space.shape[0], for non-bc
            out_features=action_space.shape[0] * 2,
            hidden_dim=hparams.actor_hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=hparams.use_actor_norm,
            use_output_norm=False,
            layers=hparams.num_actor_layers,
            hidden_skip=hparams.use_actor_skip,
            output_skip=hparams.use_actor_skip,
            rngs=rngs,
        )
    actor = Actor(
        feature_encoder=actor_encoder,
        policy_head=TanhGaussianPolicyHead(
            min_std=hparams.actor_min_std, 
            fixed_std=hparams.fixed_actor_std
            # action_low=jnp.array(action_space.low), # optional
            # action_high=jnp.array(action_space.high), # optional
        ),
        kl_start=hparams.kl_start,
        ent_start=hparams.ent_start,
        asymmetric_obs=cfg.env.get("asymmetric_observation", False),
    )
    return actor


def make_continuous_critic(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Box,
    encoder: nnx.Module = None,
    *,
    rngs: nnx.Rngs,
) -> Critic:
    hparams = cfg.algorithm
    if encoder is not None:
        feature_encoder = encoder
    else:
        feature_encoder = nnx.Sequential(
            StateActionInput(
                concatenate=True,
            ),
            MLP(
                in_features=observation_space.shape[0] + action_space.shape[0],
                out_features=hparams.critic_hidden_dim,
                hidden_dim=hparams.critic_hidden_dim,
                hidden_activation=nnx.swish,
                output_activation=None,
                use_norm=hparams.use_critic_norm,
                use_output_norm=False,
                layers=hparams.num_critic_encoder_layers,
                hidden_skip=hparams.use_critic_skip,
                output_skip=hparams.use_critic_skip,
                rngs=rngs,
            ),
        )

    q_network = nnx.Sequential(
        MLP(
            in_features=hparams.critic_hidden_dim,
            out_features=hparams.num_bins,
            hidden_dim=hparams.critic_hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=hparams.use_critic_norm,
            use_output_norm=False,
            layers=hparams.num_critic_head_layers,
            input_activation=not hparams.use_simplical_embedding,
            input_skip=hparams.use_critic_skip,
            hidden_skip=hparams.use_critic_skip,
            rngs=rngs,
        ),
        CategoricalContinuousQNetworkHead(
            num_bins=hparams.num_bins,
            vmin=hparams.vmin,
            vmax=hparams.vmax,
        ),
    )
    pred_module = MLP(
        in_features=hparams.critic_hidden_dim,
        out_features=hparams.critic_hidden_dim + 1,
        hidden_dim=hparams.critic_hidden_dim,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm=hparams.use_critic_norm,
        use_output_norm=None,
        layers=hparams.num_critic_pred_layers,
        input_activation=not hparams.use_simplical_embedding,
        input_skip=hparams.use_critic_skip,
        hidden_skip=hparams.use_critic_skip,
        output_skip=False,
        rngs=rngs,
    )
    critic = Critic(
        feature_encoder=feature_encoder,
        q_network=q_network,
        prediction_network=pred_module,
        asymmetric_obs=cfg.env.get("asymmetric_observation", False),
        is_discrete=False,
    )
    return critic


def make_discrete_actor(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Discrete,
    encoder: nnx.Module = None,
    *,
    rngs: nnx.Rngs,
) -> Actor:
    hparams = cfg.algorithm
    if encoder is not None:
        actor_encoder = encoder
        in_features = hparams.actor_hidden_dim
    else:
        actor_encoder = Identity()
        in_features = observation_space.shape[0]
    actor_head = MLP(
        in_features=in_features,
        out_features=action_space.n,
        hidden_dim=hparams.actor_hidden_dim,
        hidden_activation=nnx.relu,
        output_activation=None,
        use_norm=hparams.use_actor_norm,
        use_output_norm=False,
        layers=hparams.num_actor_layers,
        hidden_skip=hparams.use_actor_skip,
        output_skip=hparams.use_actor_skip,
        final_layer_scaling=0.01,
        rngs=rngs,
    )
    actor = Actor(
        feature_encoder=nnx.Sequential(actor_encoder, actor_head),
        policy_head=DiscretePolicyHead(),
        kl_start=hparams.kl_start,
        ent_start=hparams.ent_start,
        asymmetric_obs=cfg.env.get("asymmetric_observation", False),
    )
    return actor


def make_discrete_critic(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Discrete,
    encoder: nnx.Module = None,
    *,
    rngs: nnx.Rngs,
) -> Critic:
    hparams = cfg.algorithm
    if encoder is not None:
        feature_encoder = encoder
    else:
        feature_encoder = MLP(
            in_features=observation_space.shape[0],
            out_features=hparams.critic_hidden_dim,
            hidden_dim=hparams.critic_hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=hparams.use_critic_norm,
            use_output_norm=False,
            layers=hparams.num_critic_encoder_layers,
            hidden_skip=hparams.use_critic_skip,
            output_skip=hparams.use_critic_skip,
            rngs=rngs,
        )

    q_network = nnx.Sequential(
        StateActionInput(
            state_encoder=MLP(
                in_features=hparams.critic_hidden_dim,
                out_features=hparams.num_bins * action_space.n,
                hidden_dim=hparams.critic_hidden_dim,
                hidden_activation=nnx.swish,
                output_activation=None,
                use_norm=hparams.use_critic_norm,
                use_output_norm=False,
                layers=hparams.num_critic_head_layers,
                input_activation=False,
                input_skip=hparams.use_critic_skip,
                hidden_skip=hparams.use_critic_skip,
                rngs=rngs,
            ),
            concatenate=False,
        ),
        CategoricalDiscreteQNetworkHead(
            num_bins=hparams.num_bins,
            vmin=hparams.vmin,
            vmax=hparams.vmax,
        ),
    )
    pred_module = MLP(
        in_features=hparams.critic_hidden_dim,
        out_features=hparams.critic_hidden_dim + 1,
        hidden_dim=hparams.critic_hidden_dim,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm=hparams.use_critic_norm,
        use_output_norm=None,
        layers=hparams.num_critic_pred_layers,
        input_activation=False,
        input_skip=hparams.use_critic_skip,
        hidden_skip=hparams.use_critic_skip,
        output_skip=False,
        rngs=rngs,
    )
    critic = Critic(
        feature_encoder=feature_encoder,
        q_network=q_network,
        prediction_network=pred_module,
        asymmetric_obs=cfg.env.get("asymmetric_observation", False),
        is_discrete=True,
    )
    return critic


def make_continuous_ff_networks(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Box,
    *,
    rngs: nnx.Rngs,
) -> tuple[Actor, Critic]:
    if cfg.env.get("asymmetric_observation", False):
        q_observation_space = observation_space.spaces["privileged_state"]
        actor_observation_space = observation_space.spaces["state"]
    else:
        q_observation_space = observation_space
        actor_observation_space = observation_space

    actor = make_continuous_actor(
        cfg,
        observation_space=actor_observation_space,
        action_space=action_space,
        rngs=rngs,
    )
    critic = make_continuous_critic(
        cfg,
        observation_space=q_observation_space,
        action_space=action_space,
        rngs=rngs,
    )
    return actor, critic


def make_discrete_ff_networks(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Discrete,
    *,
    rngs: nnx.Rngs,
) -> tuple[Actor, Critic]:
    if cfg.env.get("asymmetric_observation", False):
        q_observation_space = observation_space.spaces["privileged_state"]
        actor_observation_space = observation_space.spaces["state"]
    else:
        q_observation_space = observation_space
        actor_observation_space = observation_space

    actor = make_discrete_actor(
        cfg,
        observation_space=actor_observation_space,
        action_space=action_space,
        rngs=rngs,
    )
    critic = make_discrete_critic(
        cfg,
        observation_space=q_observation_space,
        action_space=action_space,
        rngs=rngs,
    )
    return actor, critic


def make_minatar_ff_networks(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Discrete,
    *,
    rngs: nnx.Rngs,
) -> tuple[Actor, Critic]:
    cnn = MinatarConvNet(
        in_features=observation_space.shape[-1],
        normalization=nnx.LayerNorm,
        hidden_dim=cfg.algorithm.critic_hidden_dim,
        rngs=rngs,
    )
    actor_encoder = nnx.Sequential(
        nnx.clone(cnn),
        nnx.relu,
        nnx.Linear(
            in_features=cfg.algorithm.critic_hidden_dim,
            out_features=action_space.n,
            rngs=rngs,
        ),
    )
    actor = make_discrete_actor(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
        encoder=actor_encoder,
        rngs=rngs,
    )
    critic = make_discrete_critic(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
        encoder=nnx.clone(cnn),
        rngs=rngs,
    )
    return actor, critic


def make_atari_ff_networks(
    cfg: DictConfig,
    observation_space: Box,
    action_space: Discrete,
    *,
    rngs: nnx.Rngs,
) -> tuple[Actor, Critic]:
    cnn = AtariCNNEncoder(
        output_dim=cfg.algorithm.critic_hidden_dim,
        rngs=rngs,
    )
    actor = make_discrete_actor(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
        encoder=nnx.clone(cnn),
        rngs=rngs,
    )
    return actor, critic
