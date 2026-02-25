import time
import torch
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import sys
import os
import imageio
import h5py

from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax

def _compute_action_bounds(demo_path):
    """Compute action bounds from demo file once."""
    all_actions = []
    with h5py.File(demo_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith('traj_')]
        for traj_key in traj_keys:
            traj_group = f[traj_key]
            action_data = traj_group['actions'][:]
            all_actions.append(torch.tensor(action_data))
            
    all_actions = torch.cat(all_actions, dim=0)
    action_min = all_actions.min(dim=0).values
    action_max = all_actions.max(dim=0).values
    
    # Add 10% margin to bounds
    margin = 0.1 * (action_max - action_min)
    low_with_margin = action_min - margin 
    high_with_margin = action_max + margin

    # Ensure bounds are at least [-1, 1] in each dimension
    dataset_low = torch.min(low_with_margin, torch.full_like(low_with_margin, -1.0))
    dataset_high = torch.max(high_with_margin, torch.full_like(high_with_margin, 1.0))
    
    return dataset_low, dataset_high

def denormalize_action(action, dataset_low, dataset_high):
    # Denormalize action from [-1, 1] to dataset bounds.
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(np.ascontiguousarray(action.copy()))
    denormalized = (action + 1.0) * (dataset_high - dataset_low) / 2.0 + dataset_low
    return torch_to_numpy(denormalized)

def normalize_action(action, dataset_low, dataset_high):
    # Normalize action from dataset bounds to [-1, 1].
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(np.ascontiguousarray(action.copy()))
    normalized = 2.0 * (action - dataset_low) / (dataset_high - dataset_low) - 1.0
    return torch_to_numpy(normalized) 

def torch_to_numpy(tensor):
    # Convert a torch tensor to numpy, handling CUDA tensors
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    return np.array(tensor)

def get_demo_obs_keys(demo_path):
    """Extract which observation keys are actually in the demo file."""
    import h5py
    
    with h5py.File(demo_path, 'r') as f:
        traj_group = f['traj_0']
        obs_group = traj_group['obs']
        
        demo_keys = {'agent': [], 'extra': []}
        for key in sorted(obs_group.keys()):
            if key in ('agent', 'extra'):
                sub_group = obs_group[key]
                demo_keys[key] = sorted(sub_group.keys())
        
        return demo_keys

def flatten_obs(obs_dict, env, demo_obs_keys):
    # Flatten the observation dictionary to match the demo data format
    if not hasattr(obs_dict, "keys"):
        raise TypeError(
            f"flatten_obs expects a dict observation, received {type(obs_dict)}"
        )

    obs_list = []
    for key in sorted(obs_dict.keys()):
        if key in ('agent', 'extra') and key in demo_obs_keys:
            sub_group = obs_dict[key]
            for sub_key in sorted(sub_group.keys()):
                # Only include keys that were in the demo
                if sub_key in demo_obs_keys[key]:
                    data = torch_to_numpy(sub_group[sub_key])
                    data_flat = data.reshape(data.shape[0], -1) if len(data.shape) > 1 else data.reshape(1, -1)
                    obs_list.append(data_flat)
    
    # Add env state/actor vectors when the underlying env exposes them
    # state_provider = None
    # env_candidate = getattr(env, 'unwrapped', env)
    # candidate_stack = [env_candidate]
    # if hasattr(env_candidate, 'env'):
    #     candidate_stack.append(env_candidate.env)
    # if hasattr(env_candidate, 'env'):
    #     candidate_stack.append(env_candidate.env)
    # for candidate in candidate_stack:
    #     if candidate is None:
    #         continue
    #     if hasattr(candidate, 'get_state_dict'):
    #         state_provider = candidate
    #         break

    # if state_provider is not None:
    #     state = state_provider.get_state_dict()
    #     if 'actors' in state:
    #         for actor_name in sorted(state['actors'].keys()):
    #             actor_data = state['actors'][actor_name]
    #             if isinstance(actor_data, np.ndarray):
    #                 data_flat = actor_data.reshape(actor_data.shape[0], -1) if len(actor_data.shape) > 1 else actor_data.reshape(1, -1)
    #             else:
    #                 data_array = np.asarray(actor_data.cpu()) if hasattr(actor_data, 'cpu') else np.asarray(actor_data)
    #                 data_flat = data_array.reshape(1, -1) if data_array.ndim == 1 else data_array.reshape(data_array.shape[0], -1)
    #             obs_list.append(data_flat)
    
    if not obs_list:
        raise ValueError(f"No observation data found in obs_dict with keys: {obs_dict.keys()}")
    
    result = np.concatenate(obs_list, axis=1)
    return result

def make_rollout_fn(env: gymnasium.Env, num_steps: int, num_envs: int, demo_path: str = None, bc_indicator: bool = False) -> RolloutFn:
    # BC-specific rollout function with demo observation flattening
    if bc_indicator:
        demo_obs_keys = get_demo_obs_keys(demo_path) if demo_path else None
        # Compute action bounds once at function creation time
        dataset_low, dataset_high = _compute_action_bounds(demo_path)
        
        def collect_rollout(
            key: Key, train_state: TrainState, policy: Policy
        ) -> tuple[Transition, TrainState]:
            transitions = []
            obs_dict = train_state.last_obs
            obs = flatten_obs(obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            
            prev_step = train_state.time_steps
            prev_time = time.perf_counter()
            for i in range(num_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)
                
                # Convert action from JAX to numpy for environment
                action = np.asarray(action)
                # Denormalize action from [-1, 1] to dataset bounds for BC mode
                action = denormalize_action(action, dataset_low, dataset_high)
                # Get raw dict from base env
                next_obs_dict, reward, done, truncated, info = env.step(action)
                if "final_observation" in info:
                    _next_obs = to_jax(flatten_obs(info["final_observation"], env=env, demo_obs_keys=demo_obs_keys))
                else:
                    _next_obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            
                # Convert torch tensors to numpy, handling CUDA
                reward = torch_to_numpy(reward)
                done = torch_to_numpy(done)
                truncated = torch_to_numpy(truncated)
                # Normalize action back to [-1, 1] for storage in transitions
                action = normalize_action(action, dataset_low, dataset_high)
                
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
                obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)

            transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
            train_state = train_state.replace(
                last_obs=next_obs_dict,  # Store dict observation, not flattened
                time_steps=train_state.time_steps + num_steps * num_envs,
            )

            return transitions, train_state
    else:
        # Non-BC rollout function - matches upstream behavior
        def collect_rollout(
            key: Key, train_state: TrainState, policy: Policy
        ) -> tuple[Transition, TrainState]:
            transitions = []
            obs = train_state.last_obs
            prev_step = train_state.time_steps
            prev_time = time.perf_counter()
            for i in range(num_steps):
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
            transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
            train_state = train_state.replace(
                last_obs=obs,
                time_steps=train_state.time_steps + num_steps * num_envs,
            )

            return transitions, train_state

    return collect_rollout


def make_eval_fn(env: gymnasium.Env, max_episode_steps: int, demo_path: str = None, bc_indicator: bool = False) -> EvalFn:
    # BC-specific evaluation function with demo observation flattening
    if bc_indicator:
        demo_obs_keys = get_demo_obs_keys(demo_path) if demo_path else None
        # Compute action bounds once at function creation time
        dataset_low, dataset_high = _compute_action_bounds(demo_path)

        def evaluate(key: Key, policy: Policy) -> dict:
            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            
            metrics = defaultdict(list)
            num_episodes = 0
            for i in range(max_episode_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)
                # Convert action from JAX to numpy for environment
                action = np.asarray(action)
                # Denormalize action from [-1, 1] to dataset bounds for BC mode
                action = denormalize_action(action, dataset_low, dataset_high)
                # Get raw dict from base env
                next_obs_dict, reward, done, truncated, info = env.step(action)    
                if "final_info" in info:
                    mask = info["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in info["final_info"]["episode"].items():
                        metrics[k].append(v)
                obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)

            eval_metrics = {}
            for k, v in metrics.items():
                v_array = np.array([torch_to_numpy(item) for item in v])
                eval_metrics[f"{k}_std"] = v_array.std()
                eval_metrics[k] = v_array.mean()
            eval_metrics["episode_return"] = eval_metrics.pop("return", 0.0)
            eval_metrics["episode_return_std"] = eval_metrics.pop("return_std", 0.0)
            eval_metrics["episode_length"] = eval_metrics.pop("episode_len", 0.0)
            eval_metrics["episode_length_std"] = eval_metrics.pop("episode_len_std", 0.0)
            return eval_metrics
    else:
        # Non-BC evaluation function - matches upstream behavior
        def evaluate(key: Key, policy: Policy) -> dict:
            obs, _ = env.reset()
            metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(max_episode_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)
                next_obs, reward, terminated, truncated, infos = env.step(action)
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].append(v)
                obs = next_obs

            eval_metrics = {}
            for k, v in metrics.items():
                v_array = np.array([torch_to_numpy(item) for item in v])
                eval_metrics[f"{k}_std"] = v_array.std()
                eval_metrics[k] = v_array.mean()
            eval_metrics["episode_return"] = eval_metrics.pop("return", 0.0)
            eval_metrics["episode_return_std"] = eval_metrics.pop("return_std", 0.0)
            eval_metrics["episode_length"] = eval_metrics.pop("episode_len", 0.0)
            eval_metrics["episode_length_std"] = eval_metrics.pop("episode_len_std", 0.0)
            return eval_metrics

    return evaluate