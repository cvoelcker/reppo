import torch
from omegaconf import DictConfig


def make_envs(cfg: DictConfig, device: torch.device, seed: int = None) -> tuple:
    if cfg.env.type == "humanoid_bench":
        from reppo_alg.env_utils.torch_wrappers.humanoid_bench_env import (
            HumanoidBenchEnv,
        )

        envs = HumanoidBenchEnv(
            cfg.env.name, cfg.hyperparameters.num_envs, device=device
        )
        return envs, envs
    elif cfg.env.type == "isaaclab":
        from reppo_alg.env_utils.torch_wrappers.isaaclab_env import IsaacLabEnv

        envs = IsaacLabEnv(
            cfg.env.name,
            device.type,
            cfg.hyperparameters.num_envs,
            seed=seed,
            action_bounds=cfg.env.action_bounds,
        )
        return envs, envs

    elif cfg.env.type == "mjx":
        from reppo_alg.env_utils.torch_wrappers.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        envs, eval_envs = make_env(
            env_name=cfg.env.name,
            seed=seed,
            num_envs=cfg.hyperparameters.num_envs,
            num_eval_envs=cfg.hyperparameters.num_envs,
            device_rank=cfg.platform.device_rank,
            use_domain_randomization=False,
            use_push_randomization=True,
        )
        return envs, eval_envs

    elif cfg.env.type == "maniskill":
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils import gym_utils
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from reppo_alg.env_utils.torch_wrappers.maniskill_wrapper import (
            ManiSkillWrapper,
        )

        envs = gym.make(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=None,
            **cfg.env.env_kwargs,
        )
        eval_envs = gym.make(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=1,
            **cfg.env.env_kwargs,
        )
        cfg.env.max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
        # heuristic for setting gamma
        cfg.hyperparameters.gamma = 1.0 - 10.0 / cfg.env.max_episode_steps

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
        envs = ManiSkillVectorEnv(
            envs,
            cfg.hyperparameters.num_envs,
            ignore_terminations=not cfg.env.partial_reset,
            record_metrics=True,
        )
        eval_envs = ManiSkillVectorEnv(
            eval_envs,
            cfg.hyperparameters.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        return ManiSkillWrapper(
            envs,
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=cfg.env.partial_reset,
            device=device.type,
        ), ManiSkillWrapper(
            eval_envs,
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=cfg.env.partial_reset,
            device=device.type,
        )
    else:
        raise ValueError(
            f"Unknown environment type: {cfg.env.type}. Supported types are 'humanoid_bench', 'isaaclab', 'maniskill', and 'mjx'."
        )
