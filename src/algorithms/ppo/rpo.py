import logging
import math
import time

import distrax
import hydra
import jax
from flax import nnx, struct
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf
import wandb

from src.algorithms import utils
from src.algorithms.common import (
    Policy,
    TrainState,
    Transition,
    make_train_fn,
)
from src.algorithms.normalization import Normalizer
from src.algorithms.ppo.ppo import (
    PPOConfig,
    PPOTrainState,
    make_ppo_init_fn,
    ppo_policy_fn,
)

logging.basicConfig(level=logging.INFO)


class RPOConfig(PPOConfig):
    action_noise: float = 0.5


def make_rpo_learner_fn(cfg: RPOConfig):
    def loss_fn(params: nnx.Param, train_state: TrainState, minibatch: Transition):
        model = nnx.merge(train_state.graphdef, params)
        target_values = minibatch.extras["target_value"]
        advantages = minibatch.extras["advantage"]
        pi = model.actor(minibatch.obs)
        pi = distrax.MultivariateNormalDiag(
            loc=pi.mean() + minibatch.extras["action_noise"], scale_diag=pi.stddev()
        )
        value = model.critic(minibatch.obs)
        log_prob = pi.log_prob(minibatch.action)

        value_pred_clipped = minibatch.extras["value"] + (
            value - minibatch.extras["value"]
        ).clip(-cfg.clip_ratio, cfg.clip_ratio)
        value_error = jnp.square(value - target_values)
        value_error_clipped = jnp.square(value_pred_clipped - target_values)
        value_loss = 0.5 * jnp.mean(
            (1.0 - minibatch.truncated) * jnp.maximum(value_error, value_error_clipped)
        )

        ratio = jnp.exp(log_prob - minibatch.extras["log_prob"])

        actor_loss1 = ratio * advantages
        actor_loss2 = (
            jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages
        )
        actor_loss = -jnp.mean(
            (1.0 - minibatch.truncated) * jnp.minimum(actor_loss1, actor_loss2)
        )
        entropy_loss = jnp.mean(pi.entropy())

        loss = (
            actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss
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

        if cfg.normalize_advantages:
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
        key: PRNGKey, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key = jax.random.split(key)

        mini_batch_size = (
            math.floor(cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
        )
        indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size, *x.shape[1:])),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        # Compute mean metrics across mini-batches
        metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return train_state, metrics

    def learner_fn(
        key: PRNGKey, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Compute advantages and target values
        model = nnx.merge(train_state.graphdef, train_state.params)

        last_value = model.critic(train_state.last_obs)
        key, noise_key = jax.random.split(key)
        batch.extras["action_noise"] = jax.random.uniform(
            noise_key,
            batch.action.shape,
            minval=-cfg.action_noise,
            maxval=cfg.action_noise,
        )
        batch.extras["value"] = model.critic(batch.obs)
        batch.extras["log_prob"] = model.actor(batch.obs).log_prob(batch.action)

        def compute_advantage(carry, transition):
            gae, next_value = carry
            done = transition.done
            truncated = transition.truncated
            reward = transition.reward
            value = transition.extras["value"]
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.lmbda * (1 - done) * gae
            truncated_gae = reward + cfg.gamma * next_value - value
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
                (math.floor(cfg.num_steps * cfg.num_envs), *x.shape[2:])
            ),
            batch,
        )

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, data),
            init=train_state,
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    return learner_fn


@hydra.main(version_base=None, config_path="../../../config", config_name="rpo")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
        config=OmegaConf.to_container(cfg),
        name=f"rpo-{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    rpo_cfg = RPOConfig(**cfg.hyperparameters)

    # Set up the experimental environment
    env, env_params = utils.make_env(cfg)
    train_fn = make_train_fn(
        cfg=rpo_cfg,
        env=env,
        env_params=env_params,
        init_fn=make_ppo_init_fn(rpo_cfg, env, env_params),
        learner_fn=make_rpo_learner_fn(rpo_cfg),
        policy_fn=ppo_policy_fn,
        log_callback=utils.make_log_callback(),
        num_seeds=cfg.num_seeds,
    )
    start = time.perf_counter()
    _, metrics = jax.jit(train_fn)(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()
