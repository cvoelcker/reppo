import functools
import logging
import math
import time

import gymnasium
import hydra
import jax
import optax
from flax import nnx, struct
from gymnax.environments.environment import Environment, EnvParams
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf
import wandb

from src.algorithms import envs, utils
from src.common import (
    InitFn,
    Key,
    LearnerFn,
    Policy,
    TrainState,
    Transition,
)
from src.normalization import NormalizationState, Normalizer
from src.algorithms.ppo.networks import PPONetworks

logging.basicConfig(level=logging.INFO)


class PPOConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    lmbda: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    total_time_steps: int
    num_steps: int
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_advantages: bool
    normalize_env: bool
    anneal_lr: bool
    num_eval: int = 25
    max_episode_steps: int = 1000
    hidden_dim: int = 64


class PPOTrainState(TrainState):
    normalization_state: NormalizationState | None = None


def make_ppo_init_fn(
    cfg: PPOConfig, observation_space: gymnasium.Space, action_space: gymnasium.Space
) -> InitFn:
    def init(key: Key) -> PPOTrainState:
        # Number of calls to train_step
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        # Number of calls to train_iter, add 1 if not divisible by eval_interval
        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, model_key = jax.random.split(key)
        # Intialize the model
        networks = PPONetworks(
            obs_space=observation_space,
            action_space=action_space,
            hidden_dim=cfg.hidden_dim,
            rngs=nnx.Rngs(model_key),
        )

        # Set initial learning rate
        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 1e-6, num_updates)

        # Initialize the optimizer
        if cfg.max_grad_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr),
            )
        else:
            optimizer = optax.adam(lr)

        # Reset and fully initialize the environment
        key, env_key = jax.random.split(key)

        if cfg.normalize_env:
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


def make_ppo_learner_fn(cfg: PPOConfig) -> LearnerFn:
    normalizer = Normalizer()

    def loss_fn(params: nnx.Param, train_state: TrainState, minibatch: Transition):
        model = nnx.merge(train_state.graphdef, params)
        pi = model.actor(minibatch.obs)
        value = model.critic(minibatch.obs)
        log_prob = pi.log_prob(minibatch.action)
        target_values = minibatch.extras["target_value"]
        advantages = minibatch.extras["advantage"]

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
        key: Key, train_state: PPOTrainState, batch: Transition
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
        key: Key, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Compute advantages and target values
        model = nnx.merge(train_state.graphdef, train_state.params)
        last_obs = train_state.last_obs
        if cfg.normalize_env:
            norm_state = normalizer.update(train_state.normalization_state, batch.obs)
            train_state = train_state.replace(normalization_state=norm_state)
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs)
            )
            last_obs = normalizer.normalize(train_state.normalization_state, last_obs)

        last_value = model.critic(last_obs)
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

    return jax.jit(learner_fn)


def ppo_policy_fn(train_state: PPOTrainState, eval_mode: bool) -> Policy:
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

    return policy


@hydra.main(version_base=None, config_path="../../../config", config_name="ppo")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
        config=dict(cfg),
        name=f"ppo-{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    ppo_cfg = PPOConfig(**cfg.hyperparameters)

    if cfg.runner.type == "gymnax":
        from src.runners.gymnax_train_fn import make_train_fn
    elif cfg.runner.type == "gymnasium":
        from src.runners.gymnasium_train_fn import make_train_fn
    else:
        raise ValueError("Unknown environment type")

    env_setup = envs.make_env(cfg)
    train_fn = make_train_fn(
        cfg=ppo_cfg,
        env=(env_setup.env, env_setup.eval_env),
        init_fn=make_ppo_init_fn(
            cfg=ppo_cfg,
            observation_space=env_setup.observation_space,
            action_space=env_setup.action_space,
        ),
        learner_fn=make_ppo_learner_fn(ppo_cfg),
        policy_fn=ppo_policy_fn,
        log_callback=utils.make_log_callback(),
    )
    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()
