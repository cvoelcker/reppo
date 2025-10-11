import logging
import math
from typing import Callable
import operator

import hydra
import jax
import optax
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space, Box, Discrete
from src.algorithms.reppo.common import REPPOTrainState
from src.common import (
    InitFn,
    Key,
    LearnerFn,
    Policy,
    Transition,
)
from src.normalization import Normalizer
from src.algorithms import utils
import distrax

logging.basicConfig(level=logging.INFO)


class REPPOPolicy(nnx.Module):
    def __init__(
        self,
        base: nnx.Module,
        normalizer: Normalizer | None,
        normalization_state,
        eval: bool,
        action_space: Space,
    ):
        self.base = base
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self._eval_mode = eval
        self.action_space = action_space

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        if self._eval_mode:
            action = self.base.det_action(x)
        else:
            pi = self.base(x, **kwargs)
            action = pi.sample(seed=key)
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}


def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = cfg.algorithm
    offset = None

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer()
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        policy = REPPOPolicy(
            base=actor_model,
            normalizer=normalizer if cfg.normalize_env else None,
            normalization_state=train_state.normalization_state,
            eval=eval,
            action_space=action_space,
        )
        policy.eval()

        # def policy(key: Key, obs: jax.Array, **kwargs) -> tuple[jax.Array, dict]:
        #     if train_state.normalization_state is not None:
        #         obs = normalizer.normalize(train_state.normalization_state, obs)

        #     if eval:
        #         action: jax.Array = actor_model.det_action(obs)
        #     else:
        #         pi = actor_model(obs, scale=offset)
        #         action = pi.sample(seed=key)

        #     if isinstance(action_space, Box):
        #         action = action.clip(-0.999, 0.999)

        #     return action, {}

        return policy

    return policy_fn


def make_init_fn(
    cfg: DictConfig,
    observation_space: Space,
    action_space: Space,
) -> InitFn:
    hparams = cfg.algorithm
    print(action_space.shape)

    def init(key: Key):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        optim_fn = hydra.utils.instantiate(hparams.optimizer)

        if hparams.max_grad_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(hparams.max_grad_norm), optim_fn)
        else:
            tx = optim_fn

        if hparams.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(
                jax.tree.map(
                    lambda x: jnp.zeros_like(x, dtype=float),  # type: ignore
                    observation_space.sample(key),
                )
            )
        else:
            norm_state = None

        actor, critic = hydra.utils.call(cfg.algorithm.network)(
            cfg=cfg,
            action_space=action_space,
            observation_space=observation_space,
            rngs=rngs,
        )

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


def make_learner_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> LearnerFn:
    normalizer = Normalizer()
    hparams = cfg.algorithm
    discrete_actions = isinstance(action_space, Discrete)
    d = action_space.shape[-1] if not discrete_actions else action_space.n
    print(d)

    def critic_loss_fn(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_model.train()
        critic_output = critic_model(minibatch.obs, minibatch.action)

        target_values = minibatch.extras["target_values"]

        if hparams.hl_gauss:
            target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
                target_values, hparams.num_bins, hparams.vmin, hparams.vmax
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
        pred = critic_output["pred_features"]
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
        mask_truncated = hparams.mask_truncated
        mask = (1.0 - minibatch.truncated) if mask_truncated else 1.0
        loss = jnp.mean(
            mask
            * (critic_update_loss + hparams.aux_loss_mult * aux_loss)
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

        # set up models for training with batch norm
        actor_model.train()
        critic_target_model.eval()
        actor_target_model.eval()
        pi = actor_model(minibatch.obs)
        old_pi = actor_target_model(minibatch.obs)

        # policy KL constraint
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        alpha = jax.lax.stop_gradient(actor_model.temperature())
        if discrete_actions:
            critic_pred = critic_target_model(minibatch.obs)
            value = critic_pred["value"]
            actor_loss = jnp.sum(pi.probs * ((alpha * pi.logits) - value), axis=-1)
            entropy = pi.entropy()
            action_size_target = -math.log(d) * hparams.ent_target_mult
        else:
            if hparams.gradient_estimator == "score_based_gae":
                if hparams.scale_samples_with_action_d:
                    num_samples = 16 * d
                else:
                    num_samples = 16
                pred_action, aux_log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"],
                    sample_shape=(num_samples,),  # WARNING: magic number
                )
                adv = (
                    minibatch.extras["target_advs"]
                    - minibatch.extras["target_advs"].mean()
                ) / (minibatch.extras["target_advs"].std() + 1e-8)
                log_prob = pi.log_prob(minibatch.action.clip(-0.999, 0.999))
                old_log_prob = minibatch.extras["log_prob"]
                ratio = jnp.exp(log_prob - old_log_prob)
                actor_loss1 = ratio * adv
                EPS = 0.2  # hardcoded for now
                actor_loss2 = jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * adv
                actor_loss = alpha * aux_log_prob.mean(0) - jnp.minimum(
                    actor_loss1, actor_loss2
                )
                entropy = -aux_log_prob.mean(axis=0)

            elif hparams.gradient_estimator == "score_based_q":
                if hparams.scale_samples_with_action_d:
                    num_samples = 4 * d
                else:
                    num_samples = 4
                pred_action, log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"],
                    sample_shape=(num_samples,),  # WARNING: magic number
                )
                obs = jnp.repeat(minibatch.obs[None, ...], pred_action.shape[0], axis=0)
                critic_pred = critic_target_model(obs, pred_action)
                value = critic_pred["value"].sum(axis=0, keepdims=True)
                value = (value - critic_pred["value"]) / (
                    critic_pred["value"].shape[0] - 1
                )
                adv = critic_pred["value"] - value
                actor_loss = -jnp.mean(
                    log_prob * jax.lax.stop_gradient(adv) - alpha * log_prob, axis=0
                )
                entropy = -log_prob.mean(axis=0)

            elif hparams.gradient_estimator == "pathwise_q":
                pred_action, log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"]
                )
                obs = minibatch.obs
                critic_pred = critic_target_model(obs, pred_action)
                value = critic_pred["value"]
                actor_loss = log_prob * alpha - value
                entropy = -log_prob

            else:
                raise ValueError(
                    f"Unknown gradient estimator: {hparams.gradient_estimator}"
                )
            action_size_target = d * hparams.ent_target_mult

        lagrangian = actor_model.lagrangian()

        if hparams.actor_kl_clip_mode == "full":
            loss = jnp.mean(
                actor_loss + kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl
            )
        elif hparams.actor_kl_clip_mode == "clipped":
            loss = jnp.mean(
                jnp.where(
                    kl < hparams.kl_bound,
                    actor_loss,
                    kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl,
                )
            )
        elif hparams.actor_kl_clip_mode == "value":
            loss = jnp.mean(actor_loss)
        else:
            raise ValueError(f"Unknown actor loss mode: {hparams.actor_kl_clip_mode}")

        # SAC target entropy loss

        target_entropy = action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(
            target_entropy
        )

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - hparams.kl_bound)

        # total loss
        if hparams.update_entropy_lagrangian:
            loss += jnp.mean(target_entropy_loss)
        if hparams.update_kl_lagrangian:
            loss += jnp.mean(lagrangian_loss)

        # for logging
        real_action_log_prob = old_pi.log_prob(
            minibatch.action.clip(-0.999, 0.999)
        ).mean()

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
            real_action_log_prob=real_action_log_prob,
        )

    def compute_policy_kl(
        minibatch: Transition, pi: distrax.Distribution, old_pi: distrax.Distribution
    ) -> jax.Array:
        if hparams.reverse_kl:
            if discrete_actions:
                kl = pi.kl_divergence(old_pi)
            else:
                pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                    sample_shape=(4,), seed=minibatch.extras["kl_key"]
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
                    sample_shape=(4,), seed=minibatch.extras["kl_key"]
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
        grad_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
        grad_norm = jax.tree.reduce(operator.add, grad_norm)
        grads = jax.tree.map(
            lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads
        )
        actor_train_state = train_state.actor.apply_gradients(grads)
        train_state = train_state.replace(
            actor=actor_train_state,
        )
        actor_metrics = output[1]
        return train_state, {
            **critic_metrics,
            **actor_metrics,
            "grad_norm": grad_norm,
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        mini_batch_size = (
            math.floor(hparams.num_steps * hparams.num_envs) // hparams.num_mini_batches
        )
        indices = jax.random.permutation(
            shuffle_key, hparams.num_steps * hparams.num_envs
        )
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape(
                (hparams.num_mini_batches, mini_batch_size, *x.shape[1:])
            ),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)
        minibatches.extras["action_key"] = jax.random.split(
            act_key, hparams.num_mini_batches
        )
        minibatches.extras["kl_key"] = jax.random.split(
            kl_key, hparams.num_mini_batches
        )

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        # Compute mean metrics across mini-batches
        metrics_mean = jax.tree.map(lambda x: x.mean(0), metrics)
        # Compute max metrics across mini-batches
        # metrics_max = jax.tree.map(lambda x: x.max(), metrics)
        # metrics_min = jax.tree.map(lambda x: x.min(), metrics)
        return (
            train_state,
            metrics_mean,
        )  # {**metrics_mean, **{k + "_max": v for k, v in metrics_max.items()}, **{k + "_min": v for k, v in metrics_min.items()}}

    def nstep_lambda(batch: Transition):
        def loop(carry: tuple[jax.Array, ...], transition: Transition):
            lambda_return, gae, truncated, next_value = carry

            # combine importance_weights with TD lambda
            truncated = transition.truncated
            done = transition.done
            reward = transition.extras["soft_reward"]
            value = transition.extras["value"]
            policy_value = transition.extras["policy_value"]
            lambda_sum = hparams.lmbda * lambda_return + (1 - hparams.lmbda) * value
            lambda_return = reward + hparams.gamma * jnp.where(
                truncated, value, (1.0 - done) * lambda_sum
            )

            # GAE for policy
            delta = reward + hparams.gamma * (1.0 - done) * next_value - policy_value
            gae = delta + hparams.gamma * (1.0 - done) * hparams.lmbda * gae
            truncated_gae = reward + hparams.gamma * (1.0 - done) * next_value - value
            gae = jnp.where(truncated, truncated_gae, gae)

            truncated = transition.truncated
            return (
                lambda_return,
                gae,
                truncated,
                policy_value,
            ), (lambda_return, gae)

        _, (target_values, target_advs) = jax.lax.scan(
            f=loop,
            init=(
                batch.extras["value"][-1],
                batch.extras["policy_value"][-1],
                jnp.ones_like(batch.truncated[0]),
                batch.extras["policy_value"][-1],
            ),
            xs=batch,
            reverse=True,
        )
        return target_values, target_advs

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        key, act1_key, act2_key = jax.random.split(key, 3)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_model.eval()
        critic_model.eval()

        actions, log_probs = actor_model(batch.next_obs).sample_and_log_prob(
            seed=act1_key
        )
        critic_output = critic_model(batch.next_obs, actions)
        value = critic_output["value"]
        next_emb = critic_output["embed"]

        soft_reward = (
            batch.reward - hparams.gamma * log_probs * actor_model.temperature()
        )

        # compute average policy value
        if hparams.scale_samples_with_action_d:
            num_samples = 8 * d
        else:
            num_samples = 8
        pi = actor_model(batch.obs)
        actions = pi.sample(
            seed=act2_key, sample_shape=(num_samples,)
        )  # WARNING: magic number
        actions = jnp.clip(actions, -0.999, 0.999)
        obs = jnp.repeat(batch.obs[None, ...], actions.shape[0], axis=0)
        policy_value = critic_model(obs, actions)["value"].mean(0)

        extras = {
            "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
            "value": value,
            "policy_value": policy_value,
            "next_emb": next_emb,
            "log_prob": log_probs,
        }
        return extras

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        if hparams.normalize_env:
            new_norm_state = normalizer.update(
                train_state.normalization_state, batch.obs
            )
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs),
                next_obs=normalizer.normalize(
                    train_state.normalization_state, batch.next_obs
                ),
            )
            train_state = train_state.replace(normalization_state=new_norm_state)

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)

        batch.extras["target_values"], batch.extras["target_advs"] = nstep_lambda(
            batch=batch
        )

        # Reshape data to (num_steps * num_envs, ...)

        batch = jax.tree.map(
            lambda x: x.reshape((hparams.num_steps * hparams.num_envs, *x.shape[2:])),
            batch,
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
            xs=jax.random.split(train_key, hparams.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    return jax.jit(learner_fn)
