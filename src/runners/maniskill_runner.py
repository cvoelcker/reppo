import time
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import sys
import os
import imageio

from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
    
def filter_obs_to_bc_format(obs):
    """
    Extract only qpos, qvel, and tcp_pose from observation dictionary.
    Reduces 35-dim observations to 25-dim BC-compatible format.
    
    Args:
        obs: Either a dict with agent/extra keys, or already flat
    
    Returns:
        25-dimensional flattened observation array (or array of observations)
    """
    # If already flat array (not a dict), just take first 25 dims
    if isinstance(obs, (np.ndarray, jax.Array)):
        if obs.shape[-1] > 25:
            return obs[..., :25]
        return obs
    
    if isinstance(obs, dict) and 'agent' in obs:
        # obs is a dict with structure: {agent: {qpos, qvel}, extra: {tcp_pose}, ...}
        obs_list = []
        
        for key in ['agent', 'extra']:
            if key in obs and isinstance(obs[key], dict):
                for subkey in ['qpos', 'qvel', 'tcp_pose']:
                    if subkey in obs[key]:
                        subval = obs[key][subkey]
                        # Convert to array if needed
                        if isinstance(subval, (np.ndarray, jax.Array)):
                            # Reshape to (batch, -1) - assumes first dim is batch
                            if len(subval.shape) > 1:
                                flat_val = subval.reshape(subval.shape[0], -1)
                            else:
                                flat_val = subval.reshape(1, -1)
                            obs_list.append(flat_val)
        
        if obs_list:
            # Concatenate all components along the feature dimension
            filtered = np.concatenate(obs_list, axis=-1)
            return filtered
    
    # Fallback: return as-is
    return obs


def make_rollout_fn(env: gymnasium.Env, num_steps: int, num_envs: int) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment

        transitions = []
        obs = train_state.last_obs
        # obs = filter_obs_to_bc_format(obs)  # Filter observations to 25-dim format
        prev_step = train_state.time_steps
        prev_time = time.perf_counter()
        for i in range(num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(action)
            if "final_observation" in info:
                _next_obs = to_jax(info['final_observation'])
                # _next_obs = to_jax(filter_obs_to_bc_format(info["final_observation"]))
            else:
                _next_obs = next_obs
            # print('Inside Maniskill Runner:')
            # print(f'Current Step: {prev_step + i + 1}, Current Observation: {obs.shape}, Next Observation : {_next_obs.shape} Reward: {reward.shape}, Done: {done.shape}, Truncated: {truncated.shape}')
            # Record the transition
            transition = Transition(
                obs=obs,
                next_obs=_next_obs,
                action=action,
                reward=reward,
                done=done,
                truncated=truncated,
                extras={},
            )
            transitions.append(transition)
            obs = next_obs
            # if "final_info" in info:
            #     mask = info["_final_info"]
            #     print(f"Finished {mask.sum()} episodes at {train_state.time_steps/num_envs + i} sequence step with {info['final_info']['episode']['success_once'].sum()/mask.sum()} successes. Got {mask.sum()} masked and {truncated.sum()} truncation.")

        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )

        return transitions, train_state

    return collect_rollout


def make_eval_fn(env: gymnasium.Env, max_episode_steps: int) -> EvalFn:
    def evaluate(key: Key, policy: Policy, step: int = 0) -> dict:
        # Reset the environment
        obs, _ = env.reset()
        # obs = to_jax(filter_obs_to_bc_format(obs))  # Filter observations to 25-dim format
        metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(max_episode_steps):
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            next_obs, reward, terminated, truncated, infos = env.step(action)
            # next_obs = filter_obs_to_bc_format(next_obs)  # Filter observations to 25-dim format
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
