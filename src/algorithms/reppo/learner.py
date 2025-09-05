import logging
import math
from typing import Callable

import hydra
import jax
import optax
from flax import nnx, struct
from flax.struct import PyTreeNode
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space, Box
from src.common import (
    InitFn,
    Key,
    Policy,
    TrainState,
    Transition,
)
from src.normalization import Normalizer
from src.algorithms import utils
import distrax

logging.basicConfig(level=logging.INFO)


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    reverse_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    action_size_target: float = 0
    reward_scale: float = 1.0


class REPPOTrainState(TrainState):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    normalization_state: PyTreeNode | None = None


def create_exploration_offset(cfg: ReppoConfig) -> jax.Array:
    offset = (
        jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
        * (cfg.exploration_noise_max - cfg.exploration_noise_min)
        / (cfg.num_envs - cfg.exploration_base_envs)
    ) + cfg.exploration_noise_min
    offset = jnp.concatenate(
        [
            jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
            offset,
        ],
        axis=0,
    )

    return offset


def make_default_reppo_policy_fn(
    cfg: DictConfig, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = ReppoConfig(**cfg.algorithm)
    offset = create_exploration_offset(cfg)

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer()
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        def policy(key: Key, obs: jax.Array, **kwargs) -> tuple[jax.Array, dict]:
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)

            if eval:
                action: jax.Array = actor_model.det_action(obs)
            else:
                pi = actor_model.actor(obs, scale=offset)
                action = pi.sample(seed=key)

            if isinstance(action_space, Box):
                action = action.clip(-0.999, 0.999)

            return action, {}

        return policy

    return policy_fn


def make_default_init_fn(
    cfg: DictConfig,
    observation_space: Space,
    action_space: Space,
) -> InitFn:
    hparams = ReppoConfig(**cfg.algorithm)

    def init(key: Key):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        critic = hydra.utils.instantiate(cfg.networks.critic)(
            action_space=action_space,
            observation_space=observation_space,
            rngs=rngs,
        )

        actor = hydra.utils.instantiate(cfg.networks.actor)(
            action_space=action_space,
            observation_space=observation_space,
            rngs=rngs,
        )

        if not hparams.anneal_lr:
            lr = hparams.lr
        else:
            num_iterations = (
                hparams.total_time_steps // hparams.num_steps // hparams.num_envs
            )
            num_updates = num_iterations * hparams.num_epochs * hparams.num_mini_batches
            lr = optax.linear_schedule(hparams.lr, 0, num_updates)

        if hparams.max_grad_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(hparams.max_grad_norm), optax.adam(lr)
            )
        else:
            tx = optax.adam(lr)

        if hparams.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(jnp.zeros(observation_space.shape))
        else:
            norm_state = None

        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params=nnx.state(actor),
            tx=optax.set_to_zero(),
            actor=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=tx
            ),
            critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=tx
            ),
            actor_target=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor),
                params=nnx.state(actor),
                tx=optax.set_to_zero(),
            ),
            iteration=0,
            time_steps=0,
            normalization_state=norm_state,
            last_env_state=None,
            last_obs=None,
        )

    return init


def make_default_learner_fn(cfg: DictConfig, discrete_actions: bool = False):
    normalizer = Normalizer()
    cfg = ReppoConfig(**cfg.algorithm)

    def critic_loss_fn(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_output = critic_model(minibatch.obs, minibatch.action)

        target_values = minibatch.extras["target_values"]

        if cfg.hl_gauss:
            target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
                target_values, cfg.num_bins, cfg.vmin, cfg.vmax
            )
            critic_pred = critic_output["logits"]
            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        else:
            critic_pred = critic_output["value"]
            critic_update_loss = optax.squared_error(
                critic_pred.reshape(-1, 1),
                target_values.reshape(-1, 1),
            )

        # Aux loss
        pred = critic_output["embed"]
        pred_rew = critic_output["pred_rew"]
        value = critic_output["value"]
        aux_loss = optax.squared_error(pred, minibatch.extras["next_emb"])
        aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1))
            * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
        )

        # compute l2 error for logging
        critic_loss = optax.squared_error(
            value,
            target_values,
        )
        critic_loss = jnp.mean(critic_loss)
        loss = jnp.mean(
            (1.0 - minibatch.truncated)
            * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
        )
        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=aux_rew_loss,
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
        )

    def actor_loss(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_target_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_model = nnx.merge(train_state.actor.graphdef, params)
        actor_target_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor_target.params
        )
        pi = actor_model.actor(minibatch.obs)
        old_pi = actor_target_model.actor(minibatch.obs)

        # policy KL constraint
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        alpha = jax.lax.stop_gradient(actor_model.temperature())
        if discrete_actions:
            critic_pred = critic_target_model(minibatch.obs)
            value = critic_pred["value"]
            actor_loss = jnp.sum(pi.probs * ((alpha * pi.logits) - value), axis=-1)
            entropy = pi.entropy()
            action_size_target = cfg.ent_target_mult
        else:
            pred_action, log_prob = pi.sample_and_log_prob(
                seed=minibatch.extras["action_key"]
            )
            critic_pred = critic_target_model(minibatch.obs, pred_action)
            value = critic_pred["value"]
            actor_loss = log_prob * alpha - value
            entropy = -log_prob
            action_size_target = pred_action.shape[-1] * cfg.ent_target_mult

        lagrangian = actor_model.lagrangian()

        if cfg.actor_kl_clip_mode == "full":
            loss = jnp.mean(
                actor_loss + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
            )
        elif cfg.actor_kl_clip_mode == "clipped":
            loss = jnp.mean(
                jnp.where(
                    kl < cfg.kl_bound,
                    actor_loss,
                    kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                )
            )
        elif cfg.actor_kl_clip_mode == "value":
            loss = jnp.mean(actor_loss)
        else:
            raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

        # SAC target entropy loss

        target_entropy = action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(
            target_entropy
        )

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - cfg.kl_bound)

        # total loss
        if cfg.update_entropy_lagrangian:
            loss += jnp.mean(target_entropy_loss)
        if cfg.update_kl_lagrangian:
            loss += jnp.mean(lagrangian_loss)

        return loss, dict(
            actor_loss=actor_loss,
            loss=loss,
            temp=actor_model.temperature(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean()
            if not discrete_actions
            else 0.0,
            reward_mean=minibatch.reward.mean(),
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            entropy=entropy,
            entropy_loss=target_entropy_loss,
            target_values=minibatch.extras["target_values"].mean(),
        )

    def compute_policy_kl(
        minibatch: Transition, pi: distrax.Distribution, old_pi: distrax.Distribution
    ) -> jax.Array:
        if cfg.reverse_kl:
            if discrete_actions:
                kl = pi.kl_divergence(old_pi)
            else:
                pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                    sample_shape=(16,), seed=minibatch.extras["kl_key"]
                )
                pi_action = jnp.clip(pi_action, -1 + 1e-4, 1 - 1e-4)
                old_pi_act_log_prob = old_pi.log_prob(pi_action).mean(0)
                pi_act_log_prob = pi_act_log_prob.mean(0)
                kl = pi_act_log_prob - old_pi_act_log_prob
        else:
            if discrete_actions:
                kl = old_pi.kl_divergence(pi)
            else:
                old_pi_action, old_pi_act_log_prob = old_pi.sample_and_log_prob(
                    sample_shape=(16,), seed=minibatch.extras["kl_key"]
                )
                old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)

                old_pi_act_log_prob = old_pi_act_log_prob.mean(0)
                pi_act_log_prob = pi.log_prob(old_pi_action).mean(0)
                kl = old_pi_act_log_prob - pi_act_log_prob
        return kl

    def update(train_state: REPPOTrainState, batch: Transition):
        # Sample data at indices from the batch
        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
        critic_train_state = train_state.critic.apply_gradients(grads)
        train_state = train_state.replace(
            critic=critic_train_state,
        )
        critic_metrics = output[1]

        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
        actor_train_state = train_state.actor.apply_gradients(grads)
        train_state = train_state.replace(
            actor=actor_train_state,
        )
        actor_metrics = output[1]
        return train_state, {
            **critic_metrics,
            **actor_metrics,
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        mini_batch_size = (
            math.floor(cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
        )
        indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size, *x.shape[1:])),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)
        minibatches.extras["action_key"] = jax.random.split(
            act_key, cfg.num_mini_batches
        )
        minibatches.extras["kl_key"] = jax.random.split(kl_key, cfg.num_mini_batches)

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        # Compute mean metrics across mini-batches
        metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return train_state, metrics

    def nstep_lambda(batch: Transition):
        def loop(carry: tuple[jax.Array, ...], transition: Transition):
            lambda_return, truncated, importance_weight = carry

            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.extras["soft_reward"]
            value = transition.extras["value"]
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.extras["importance_weight"],
            ), lambda_return

        _, target_values = jax.lax.scan(
            f=loop,
            init=(
                batch.extras["value"][-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.extras["importance_weight"][0]),
            ),
            xs=batch,
            reverse=True,
        )
        return target_values

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        offset = create_exploration_offset(cfg=cfg)

        last_obs = train_state.last_obs
        if cfg.normalize_env:
            last_obs = normalizer.normalize(train_state.normalization_state, last_obs)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        critic_output = critic_model(batch.obs, batch.action)
        emb = critic_output["embed"]
        value = critic_output["value"]
        og_pi = actor_model.actor(batch.obs)
        pi = actor_model.actor(batch.obs, scale=offset)
        key, act_key = jax.random.split(key)

        last_action, last_log_prob = actor_model.actor(last_obs).sample_and_log_prob(
            seed=act_key
        )
        last_critic_output = critic_model(last_obs, last_action)
        last_emb = last_critic_output["embed"]
        last_value = last_critic_output["value"]

        log_probs = pi.log_prob(batch.action)
        next_log_prob = jnp.concatenate([log_probs[1:], last_log_prob[None]], axis=0)

        soft_reward = (
            batch.reward - cfg.gamma * next_log_prob * actor_model.temperature()
        )

        raw_importance_weight = jnp.nan_to_num(
            og_pi.log_prob(batch.action) - pi.log_prob(batch.action),
            nan=jnp.log(cfg.lmbda_min),
        )
        importance_weight = jnp.clip(
            raw_importance_weight, min=jnp.log(cfg.lmbda_min), max=jnp.log(1.0)
        )

        extras = {
            "soft_reward": soft_reward * cfg.reward_scale,
            "value": jnp.concatenate([value[1:], last_value[None]], axis=0),
            "next_emb": jnp.concatenate([emb[1:], last_emb[None]], axis=0),
            "importance_weight": importance_weight,
        }
        return extras

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        if cfg.normalize_env:
            new_norm_state = normalizer.update(
                train_state.normalization_state, batch.obs
            )
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs)
            )
            train_state = train_state.replace(normalization_state=new_norm_state)

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)

        batch.extras["target_values"] = nstep_lambda(batch=batch)

        # Reshape data to (num_steps * num_envs, ...)

        batch = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), batch
        )
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch),
            init=train_state,
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    return learner_fn
