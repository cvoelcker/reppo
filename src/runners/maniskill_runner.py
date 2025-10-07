from collections import defaultdict
import gymnasium
import jax
import numpy as np
from src.common import (
    EvalFn,
    Key,
    Policy,
)


def make_eval_fn(env: gymnasium.Env, max_episode_steps: int) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Reset the environment
        obs, _ = env.reset()
        metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(max_episode_steps):
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            next_obs, reward, terminated, truncated, infos = env.step(np.array(action))
            if "final_info" in infos:
                mask = infos["_final_info"]
                num_episodes += mask.sum()
                for k, v in infos["final_info"]["episode"].items():
                    metrics[k].append(v)
            obs = next_obs

        eval_metrics = {}
        for k, v in metrics.items():
            eval_metrics[f"{k}_std"] = np.array(v).std()
            eval_metrics[k] = np.array(v).mean()
        eval_metrics["episode_return"] = eval_metrics.pop("return", 0.0)
        eval_metrics["episode_return_std"] = eval_metrics.pop("return_std", 0.0)
        eval_metrics["episode_length"] = eval_metrics.pop("episode_len", 0.0)
        eval_metrics["episode_length_std"] = eval_metrics.pop("episode_len_std", 0.0)
        return eval_metrics

    return evaluate
