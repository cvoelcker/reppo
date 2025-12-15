import time
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from collections import defaultdict

from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)
from src.algorithms import utils
from src.env_utils.torch_wrappers.torch_cuda_wrapper import to_jax

logger = utils.setup_logger("reppo/torch_cuda_runner")

def make_rollout_fn(env: gymnasium.Env, num_steps: int, num_envs: int) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment

        transitions = []
        obs = train_state.last_obs
        prev_step = train_state.time_steps
        prev_time = time.perf_counter()
        for i in range(num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(action)
            if "final_observation" in info:
                _next_obs = to_jax(info["final_observation"])
            else:
                _next_obs = next_obs
            # Record the transition
            transition = Transition(
                obs=obs.copy(),
                action=action.copy(),
                reward=reward.copy(),
                done=done.copy(),
                truncated=truncated.copy(),
                extras={},
            )
            transitions.append(transition)
            obs = next_obs
        logger.info(f'Current performance: {info["log_info"]["return"].mean().item():.2f} return in {info["log_info"]["episode_len"].mean().item()} steps.')
        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )

        return transitions, train_state, info["log_info"]

    return collect_rollout


def make_eval_fn(env: gymnasium.Env, max_episode_steps: int) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        print(f"RUNNING FOR {max_episode_steps} STEPS")
        # Reset the environment
        obs, _ = env.reset(random_start_init=False)
        num_episodes = 0
        all_done = jnp.zeros(env.num_envs, dtype=jnp.bool_)
        for i in range(max_episode_steps):
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            next_obs, reward, terminated, truncated, infos = env.step(action)
            obs = next_obs
            all_done = all_done | terminated | truncated
            if jnp.all(all_done):
                break
        eval_metrics = {}
        for k, v in infos["log_info"].items():
            eval_metrics[f"{k}_std"] = np.array(v).std()
            eval_metrics[k] = np.array(v).mean()
        obs, _ = env.reset(random_start_init=True)
        return eval_metrics, obs

    return evaluate