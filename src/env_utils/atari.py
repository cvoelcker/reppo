import gymnasium as gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100, is_legacy_gym=False):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        # get if the env has lives
        self.has_lives = False
        self.is_legacy_gym = is_legacy_gym
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")

    def reset(self, **kwargs):
        if self.is_legacy_gym:
            observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        else:
            observations, infos = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, infos if not self.is_legacy_gym else observations

    def step(self, action):
        if self.is_legacy_gym:
            observations, rewards, dones, infos = super(
                RecordEpisodeStatistics, self
            ).step(action)
        else:
            observations, rewards, term, trunc, infos = super(
                RecordEpisodeStatistics, self
            ).step(action)
            dones = term + trunc
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        all_lives_exhausted = infos["lives"] == 0
        if self.has_lives:
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            np.zeros_like(dones, dtype=bool),
            infos,
        )
