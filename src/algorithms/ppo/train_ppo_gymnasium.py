import logging
import time

import gymnasium
import hydra
import jax
import optax
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf
import wandb

from src.algorithms import utils
from src.algorithms.common import (
    InitFn,
    Key,
    make_gymnasium_eval_fn,
    make_gymnasium_rollout_fn,
    make_train_fn,
)
from src.algorithms.normalization import Normalizer
from src.algorithms.ppo.networks import PPONetworks
from src.algorithms.ppo.ppo import (
    PPOConfig,
    PPOTrainState,
    make_ppo_learner_fn,
    ppo_policy_fn,
)

logging.basicConfig(level=logging.INFO)


def make_ppo_init_fn(
    cfg: PPOConfig,
    observation_space: gymnasium.spaces.Space,
    action_space: gymnasium.spaces.Space,
) -> InitFn:
    def init(key: Key) -> PPOTrainState:
        # Number of calls to train_step

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

        if cfg.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(
                jax.tree.map(jnp.zeros_like, observation_space.sample())
            )
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


def make_env_fn(cfg: DictConfig):
    def make_env():
        env = gymnasium.make(cfg.env.name)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        return env

    return make_env


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

    # Set up the experimental environment
    env = gymnasium.vector.SyncVectorEnv([make_env_fn(cfg) for _ in range(ppo_cfg.num_envs)])
    eval_env = gymnasium.vector.SyncVectorEnv([make_env_fn(cfg) for _ in range(ppo_cfg.num_envs)])

    train_fn = make_train_fn(
        cfg=ppo_cfg,
        env=(env, eval_env),
        init_fn=make_ppo_init_fn(
            ppo_cfg, env.single_observation_space, env.single_action_space
        ),
        rollout_fn=make_gymnasium_rollout_fn(ppo_cfg, env),
        eval_fn=make_gymnasium_eval_fn(ppo_cfg, eval_env),
        learner_fn=jax.jit(make_ppo_learner_fn(ppo_cfg)),
        policy_fn=ppo_policy_fn,
        log_callback=utils.make_log_callback(multiple_seeds=False),
        num_seeds=1,
        mode="loop",
    )
    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()
