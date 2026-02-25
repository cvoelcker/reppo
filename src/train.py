import hydra
import jax
import logging
import time
import wandb
from omegaconf import DictConfig, OmegaConf
from src.algorithms import envs, utils
from src.common import InitFn, LearnerFn, PolicyFn
from src.cfg_utils import fix_cfg
import jax.numpy as jnp
from gymnasium import spaces
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
import torch

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
    
    # Modify run name based on bc_indicator
    bc_suffix = f"-bc-denorm-{cfg.algorithm.bc_actor_update_delay}" if cfg.algorithm.bc_indicator else ""
    run_name = f"{cfg.name}-{cfg.env.name.lower()}{bc_suffix}"
    
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=cfg.tags,
        config=OmegaConf.to_container(cfg),
        name=run_name,
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    
    if cfg.algorithm.bc_indicator:
        # Load dataset first to get observation dimension (dataset dims) like test_bc.py does
        logging.info(f"Loading dataset from {cfg.env.demo.demo_path}")
        config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
        loader = ManiSkillDemoLoader(config, cfg.env.name)
        trajectories, _ = loader.load_demo_dataset(cfg.env.demo.demo_path)
        n_obs_dataset = trajectories[0]['observations'].shape[1]
        logging.info(f"Dataset observation dimension: {n_obs_dataset}")
        # Create environment with the correct observation dimension
        env_setup = envs.make_env(cfg, n_obs_dataset=n_obs_dataset)
        obs_space = env_setup.observation_space
    else:
        env_setup = envs.make_env(cfg)
        obs_space = env_setup.observation_space
    
    init_fn: InitFn = hydra.utils.call(cfg.algorithm.init)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    learner_fn: LearnerFn = hydra.utils.call(cfg.algorithm.learner)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    policy_fn: PolicyFn = hydra.utils.call(cfg.algorithm.policy)(
        cfg=cfg,
        action_space=env_setup.action_space,
        observation_space=obs_space,
    )
    rollout_fn = hydra.utils.call(cfg.runner.rollout_fn)(env_setup.env, demo_path=cfg.env.demo.demo_path, bc_indicator=cfg.algorithm.bc_indicator)
    eval_fn = hydra.utils.call(cfg.runner.eval_fn)(env_setup.eval_env, demo_path=cfg.env.demo.demo_path, bc_indicator=cfg.algorithm.bc_indicator)
    make_train_fn = hydra.utils.call(cfg.runner.train_fn)
    train_fn = make_train_fn(
        env=(env_setup.env, env_setup.eval_env),
        init_fn=init_fn,
        learner_fn=learner_fn,
        policy_fn=policy_fn,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
        log_callback=utils.make_log_callback(),
        demo_path=cfg.env.demo.demo_path,
        bc_indicator=cfg.algorithm.bc_indicator
    )
    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()
