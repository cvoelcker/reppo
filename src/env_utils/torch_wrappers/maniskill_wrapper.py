from gymnasium import Wrapper
import torch


class ManiSkillWrapper(Wrapper):
    """
    A wrapper for ManiSkill environments to ensure compatibility with the expected API.
    This wrapper is used to handle the ManiSkill environments in a way that is consistent
    with the other environments in the codebase.
    """

    def __init__(self, env, max_episode_steps: int, partial_reset, device: str):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.asymmetric_obs = False
        self.max_episode_steps = max_episode_steps

        self.partial_reset = partial_reset

        self.returns = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
        self.episode_len = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
        self.success = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    @property
    def unwrapped(self):
        """
        Returns the underlying environment.
        """
        return self.env

    @property
    def num_actions(self):
        """
        Returns the number of actions in the action space.
        """
        return self.action_space.shape[1]

    @property
    def num_obs(self):
        """
        Returns the number of observations in the observation space.
        """
        return self.observation_space.shape[1]

    def reset(self, seed=None, options=dict()):
        """
        Resets the environment and returns the initial observation.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        """
        Takes a step in the environment with the given action.
        Returns the next observation, reward, done, and info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "final_info" in info:
            self.returns = (
                info["final_info"]["episode"]["return"] * info["_final_info"].float()
                + (1.0 - info["_final_info"].float()) * self.returns
            )
            self.episode_len = (
                info["final_info"]["episode"]["episode_len"]
                * info["_final_info"].float()
                + (1.0 - info["_final_info"].float()) * self.episode_len
            )
            self.success = (
                info["final_info"]["episode"]["success_once"]
                * info["_final_info"].float()
                + (1.0 - info["_final_info"].float()) * self.success
            )
        info["log_info"] = {
            "return": self.returns,
            "episode_len": self.episode_len,
            "success": self.success,
        }
        if self.partial_reset:
            # maniskill continues bootstrap on terminated, which playground does on truncated.
            # This unifies the interfaces in a very hacky way
            done = torch.zeros_like(
                terminated, dtype=torch.bool, device=terminated.device
            )
            truncated = torch.logical_or(terminated, truncated)
        else:
            done = torch.logical_or(terminated, truncated)
            truncated = torch.zeros_like(done, dtype=torch.bool, device=done.device)
        return obs, reward, done, truncated, info
