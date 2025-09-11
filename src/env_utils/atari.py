import gymnasium as gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = env.config["num_envs"]
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        observations, rewards, dones, truncated, infos = self.env.step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        for i in np.argwhere(infos["terminated"] & (infos["lives"] == 0)).squeeze(1):
            if "final_info" not in infos:
                infos["final_info"] = []
            infos["final_info"].append({
                "episode": {
                    "r": self.returned_episode_returns[i],
                    "l": self.returned_episode_lengths[i],
                }
            })
        return (
            observations,
            rewards,
            dones,
            truncated,
            infos,
        )

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_observation_space(self):
        return self.env.observation_space

    @property
    def single_action_space(self):
        return self.env.action_space
