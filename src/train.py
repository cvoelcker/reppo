import hydra
import jax
import logging
import time
import wandb
from omegaconf import DictConfig, OmegaConf
from src.algorithms import envs, utils
from src.common import InitFn, LearnerFn, PolicyFn
from src.cfg_utils import fix_cfg

logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../config/default",
    config_name="reppo_continuous.yaml",
)
def main(cfg: DictConfig):
    cfg = fix_cfg(cfg)
    OmegaConf.resolve(cfg)
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=cfg.tags,
        config=OmegaConf.to_container(cfg),
        name=f"{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    env_setup = envs.make_env(cfg)
    init_fn: InitFn = hydra.utils.call(cfg.algorithm.init)(
        cfg=cfg,
        observation_space=env_setup.observation_space,
        action_space=env_setup.action_space,
    )
    learner_fn: LearnerFn = hydra.utils.call(cfg.algorithm.learner)(
        cfg=cfg,
        observation_space=env_setup.observation_space,
        action_space=env_setup.action_space,
    )
    policy_fn: PolicyFn = hydra.utils.call(cfg.algorithm.policy)(
        cfg=cfg,
        action_space=env_setup.action_space,
        observation_space=env_setup.observation_space,
    )
    rollout_fn = hydra.utils.call(cfg.runner.rollout_fn)(env=env_setup.env)
    eval_fn = hydra.utils.call(cfg.runner.eval_fn)(env=env_setup.eval_env)
    make_train_fn = hydra.utils.call(cfg.runner.train_fn)
    train_fn = make_train_fn(
        env=(env_setup.env, env_setup.eval_env),
        init_fn=init_fn,
        learner_fn=learner_fn,
        policy_fn=policy_fn,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
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
