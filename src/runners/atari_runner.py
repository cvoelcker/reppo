from collections import defaultdict
import gymnasium
import jax
import numpy as np
from src.common import (
    EvalFn,
    Key,
    Policy,
)


def make_eval_fn(
    env: gymnasium.Env, max_episode_steps: int, max_eval_episodes: int
) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Evaluate the policy in the environment
        key, eval_key = jax.random.split(key)

        # Reset the environment
        obs, _ = env.reset()
        done = False
        dones = []
        episode_rewards = []
        episode_lengths = []
        num_episodes = 0
        for _ in range(max_episode_steps):
            # Select action
            action, _ = policy(eval_key, obs)
            # Step the environment
            next_obs, reward, done, truncated, info = env.step(np.array(action))
            obs = next_obs
            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    episode_rewards.append(info["r"][idx])
                    episode_lengths.append(info["l"][idx])
                    num_episodes += 1
            if num_episodes >= max_eval_episodes:
                break

        return {
            "episode_return": np.mean(episode_rewards) if episode_rewards else 0.0,
            "episode_return_std": np.std(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": len(episode_rewards),
            "episode_lengths": episode_lengths,
        }
    return evaluate