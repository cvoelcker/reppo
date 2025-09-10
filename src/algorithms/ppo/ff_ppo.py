import logging
import math

import distrax
import gymnasium
import hydra
import jax
import optax
from flax import nnx, struct
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space

from src.common import (
    InitFn,
    Key,
    LearnerFn,
    Policy,
    TrainState,
    Transition,
)
from src.normalization import NormalizationState, Normalizer

logging.basicConfig(level=logging.INFO)


class PPOTrainState(TrainState):
    normalization_state: NormalizationState | None = None


def make_init_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> InitFn:
    algo_cfg = cfg.algorithm

    def init(key: Key) -> PPOTrainState:
        # Number of calls to train_step
        num_train_steps = algo_cfg.total_time_steps // (
            algo_cfg.num_steps * algo_cfg.num_envs
        )
        # Number of calls to train_iter, add 1 if not divisible by eval_interval
        eval_interval = int(
            (algo_cfg.total_time_steps / (algo_cfg.num_steps * algo_cfg.num_envs))
            // algo_cfg.num_eval
        )
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, model_key = jax.random.split(key)
        # Intialize the model
        networks = hydra.utils.instantiate(cfg.algorithm.network)(
            obs_space=observation_space,
            action_space=action_space,
            rngs=nnx.Rngs(model_key),
        )

        # Set initial learning rate
        if not algo_cfg.anneal_lr:
            lr = algo_cfg.lr
        else:
            num_iterations = (
                algo_cfg.total_time_steps // algo_cfg.num_steps // algo_cfg.num_envs
            )
            num_updates = (
                num_iterations * algo_cfg.num_epochs * algo_cfg.num_mini_batches
            )
            lr = optax.linear_schedule(algo_cfg.lr, 1e-6, num_updates)

        # Initialize the optimizer
        if algo_cfg.max_grad_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(algo_cfg.max_grad_norm),
                optax.adam(lr),
            )
        else:
            optimizer = optax.adam(lr)

        # Reset and fully initialize the environment
        key, env_key = jax.random.split(key)

        if algo_cfg.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(jnp.zeros(observation_space.shape))
        else:
            norm_state = None

        # Initialize the state observations of the environment
        return PPOTrainState.create(
            iteration=0,
            time_steps=0,
            graphdef=nnx.graphdef(networks),
            params=nnx.state(networks),
            tx=optimizer,
            last_env_state=None,
            last_obs=None,
            normalization_state=norm_state,
        )

    return init


def make_learner_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> LearnerFn:
    algo_cfg = cfg.algorithm
    normalizer = Normalizer()

    def loss_fn(params: nnx.Param, train_state: TrainState, minibatch: Transition):
        model = nnx.merge(train_state.graphdef, params)
        pi = model.actor(minibatch.obs)

        if algo_cfg.loss == "rpo":
            pi = distrax.MultivariateNormalDiag(
                loc=pi.mean() + minibatch.extras["action_noise"], scale_diag=pi.stddev()
            )

        value = model.critic(minibatch.obs)
        log_prob = pi.log_prob(minibatch.action)
        target_values = minibatch.extras["target_value"]
        advantages = minibatch.extras["advantage"]

        value_pred_clipped = minibatch.extras["value"] + (
            value - minibatch.extras["value"]
        ).clip(-algo_cfg.clip_ratio, algo_cfg.clip_ratio)
        value_error = jnp.square(value - target_values)
        value_error_clipped = jnp.square(value_pred_clipped - target_values)
        value_loss = 0.5 * jnp.mean(
            (1.0 - minibatch.truncated) * jnp.maximum(value_error, value_error_clipped)
        )

        if algo_cfg.loss == "dpo":
            log_diff = log_prob - minibatch.extras["log_prob"]
            ratio = jnp.exp(log_diff)
            r1 = ratio - 1.0
            drift1 = jax.nn.relu(
                r1 * advantages
                - algo_cfg.alpha * jax.nn.tanh(r1 * advantages / algo_cfg.alpha)
            )
            drift2 = jax.nn.relu(
                log_diff * advantages
                - algo_cfg.beta * jax.nn.tanh(log_diff * advantages / algo_cfg.beta)
            )
            drift = jnp.where(
                advantages >= 0.0,
                drift1,
                drift2,
            )
            losses = ratio * advantages - drift
            mask = 1.0 - minibatch.truncated
            actor_loss = -jnp.mean(losses * mask)
            entropy_loss = jnp.mean(pi.entropy())

            loss = (
                actor_loss
                + algo_cfg.value_coef * value_loss
                - algo_cfg.entropy_coef * entropy_loss
            )
        else:
            ratio = jnp.exp(log_prob - minibatch.extras["log_prob"])

            actor_loss1 = ratio * advantages
            actor_loss2 = (
                jnp.clip(ratio, 1 - algo_cfg.clip_ratio, 1 + algo_cfg.clip_ratio)
                * advantages
            )
            actor_loss = -jnp.mean(
                (1.0 - minibatch.truncated) * jnp.minimum(actor_loss1, actor_loss2)
            )
            entropy_loss = jnp.mean(pi.entropy())

            loss = (
                actor_loss
                + algo_cfg.value_coef * value_loss
                - algo_cfg.entropy_coef * entropy_loss
            )

        return loss, dict(
            actor_loss=actor_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            loss=loss,
            mean_value=value.mean(),
            mean_log_prob=log_prob.mean(),
            mean_advantages=advantages.mean(),
            mean_action=minibatch.action.mean(),
            mean_reward=minibatch.reward.mean(),
        )

    def update(train_state: PPOTrainState, batch: Transition):
        # Sample data at indices from the batch

        if algo_cfg.normalize_advantages:
            advantages = batch.extras["advantage"]
            batch.extras["advantage"] = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + 1e-8
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        output, grads = grad_fn(train_state.params, train_state, batch)

        # Global gradient norm (all parameters combined)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        global_grad_norm = jnp.linalg.norm(flat_grads)

        metrics = output[1]
        metrics["advantages"] = batch.extras["advantage"]
        metrics["global_grad_norm"] = global_grad_norm
        train_state = train_state.apply_gradients(grads)
        return train_state, metrics

    def run_epoch(
        key: Key, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key = jax.random.split(key)

        mini_batch_size = (
            math.floor(algo_cfg.num_steps * algo_cfg.num_envs)
            // algo_cfg.num_mini_batches
        )
        indices = jax.random.permutation(
            shuffle_key, algo_cfg.num_steps * algo_cfg.num_envs
        )
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape(
                (algo_cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
            ),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        # Compute mean metrics across mini-batches
        metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return train_state, metrics

    def learner_fn(
        key: Key, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Compute advantages and target values
        model = nnx.merge(train_state.graphdef, train_state.params)
        last_obs = train_state.last_obs
        if algo_cfg.normalize_env:
            norm_state = normalizer.update(train_state.normalization_state, batch.obs)
            train_state = train_state.replace(normalization_state=norm_state)
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs)
            )
            last_obs = normalizer.normalize(train_state.normalization_state, last_obs)

        last_value = model.critic(last_obs)
        batch.extras["value"] = model.critic(batch.obs)
        batch.extras["log_prob"] = model.actor(batch.obs).log_prob(batch.action)
        if algo_cfg.loss == "rpo":
            key, noise_key = jax.random.split(key)
            batch.extras["action_noise"] = jax.random.uniform(
                noise_key,
                batch.action.shape,
                minval=-algo_cfg.action_noise,
                maxval=algo_cfg.action_noise,
            )

        def compute_advantage(carry, transition):
            gae, next_value = carry
            done = transition.done
            truncated = transition.truncated
            reward = transition.reward
            value = transition.extras["value"]
            delta = reward + algo_cfg.gamma * next_value * (1 - done) - value
            gae = delta + algo_cfg.gamma * algo_cfg.lmbda * (1 - done) * gae
            truncated_gae = reward + algo_cfg.gamma * next_value - value
            gae = jnp.where(truncated, truncated_gae, gae)
            return (gae, value), gae

        # Compute the advantage using GAE
        _, advantages = jax.lax.scan(
            compute_advantage,
            (jnp.zeros_like(last_value), last_value),
            batch,
            reverse=True,
        )
        target_values = advantages + batch.extras["value"]
        batch.extras["advantage"] = advantages
        batch.extras["target_value"] = target_values

        # Reshape data to (num_steps * num_envs, ...)
        data = jax.tree.map(
            lambda x: x.reshape(
                (math.floor(algo_cfg.num_steps * algo_cfg.num_envs), *x.shape[2:])
            ),
            batch,
        )

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, data),
            init=train_state,
            xs=jax.random.split(train_key, algo_cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    return jax.jit(learner_fn)


def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Policy:
    def policy_fn(train_state: PPOTrainState, eval_mode: bool) -> Policy:
        normalizer = Normalizer()

        def policy(
            key: Key, obs: jax.Array, state: struct.PyTreeNode | None = None
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)
            model = nnx.merge(train_state.graphdef, train_state.params)
            pi = model.actor(obs)
            value = model.critic(obs)
            action = pi.sample(seed=key)
            log_prob = pi.log_prob(action)
            return action, dict(log_prob=log_prob, value=value)

        return jax.jit(policy)

    return policy_fn
