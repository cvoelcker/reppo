import hydra
import jax
import logging
import time
import wandb
from omegaconf import DictConfig, OmegaConf
from gymnax.environments.spaces import Discrete
from src.algorithms.reppo.learner import (
    make_default_init_fn,
    make_default_learner_fn,
)
from src.algorithms.reppo.learner import make_default_reppo_policy_fn
from src.algorithms import envs, utils


@hydra.main(version_base=None, config_path="../../../config/default/reppo", config_name="ff_playground.yaml")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
        config=OmegaConf.to_container(cfg),
        name=f"reppo-{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    env_setup = envs.make_env(cfg)
    make_train_fn = hydra.utils.call(cfg.runner)
    train_fn = make_train_fn(
        env=(env_setup.env, env_setup.eval_env),
        init_fn=make_default_init_fn(
            cfg=cfg,
            observation_space=env_setup.observation_space,
            action_space=env_setup.action_space,
        ),
        learner_fn=make_default_learner_fn(
            cfg, discrete_actions=isinstance(env_setup.action_space, Discrete)
        ),
        policy_fn=make_default_reppo_policy_fn(cfg, env_setup.action_space),
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
