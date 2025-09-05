from collections import defaultdict
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from src.algorithms import utils
from src.common import (
    Config,
    EvalFn,
    InitFn,
    Key,
    LearnerFn,
    LogCallback,
    Policy,
    PolicyFn,
    RolloutFn,
    TrainState,
    Transition,
)
from src.runners.gymnasium_train_fn import make_train_fn as make_gymnasium_train_fn


def make_train_fn(
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    max_episode_steps: int,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
):
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    return make_gymnasium_train_fn(
        env=env,
        total_time_steps=total_time_steps,
        num_steps=num_steps,
        num_envs=num_envs,
        num_eval=num_eval,
        max_episode_steps=max_episode_steps,
        init_fn=init_fn,
        policy_fn=policy_fn,
        learner_fn=learner_fn,
        rollout_fn=rollout_fn,
        eval_fn=make_maniskill_eval_fn(
            eval_env, num_eval_steps=max(max_episode_steps, num_steps)
        )
        if eval_fn is None
        else eval_fn,
        log_callback=log_callback,
    )


def make_maniskill_eval_fn(env: gymnasium.Env, num_eval_steps: int) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Reset the environment
        obs, _ = env.reset()
        metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(num_eval_steps):
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
