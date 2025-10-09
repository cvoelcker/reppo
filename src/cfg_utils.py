from omegaconf import DictConfig
from omegaconf import OmegaConf


def fix_cfg(cfg: DictConfig) -> DictConfig:

    # Convert to standard dict and back to resolve interpolations
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg_dict["env"]["type"] == "maniskill":
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils import gym_utils
        env_kwargs = cfg.env.kwargs if "kwargs" in cfg.env else {}
        if cfg.env.control_mode is not None:
            env_kwargs["control_mode"] = cfg.env.control_mode
        reconfiguration_freq = (
            None
        )
        envs = gym.make(
            cfg.env.name,
            num_envs=cfg.algorithm.num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **env_kwargs,
        )
        cfg_dict["env"]["max_episode_steps"] = gym_utils.find_max_episode_steps_value(envs)

    return OmegaConf.create(cfg_dict)