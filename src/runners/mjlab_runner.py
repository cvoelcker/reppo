import time
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)
import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv


def torch_to_jax_array(x: torch.Tensor):
    def map_fn(array):
        return jax.dlpack.from_dlpack(array)

    return jax.tree.map(map_fn, x)


def jax_to_torch_array(x: jax.Array):
    def map_fn(array):
        return torch.utils.dlpack.from_dlpack(array)

    return jax.tree.map(map_fn, x)


def make_rollout_fn(env: gymnasium.Env, num_steps: int, num_envs: int) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment

        transitions = []
        obs = train_state.last_obs
        prev_step = train_state.time_steps
        prev_time = time.perf_counter()
        for _ in range(num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(
                jax_to_torch_array(action)
            )
            # Record the transition
            transition = Transition(
                obs=obs,
                action=action,
                reward=torch_to_jax_array(reward),
                done=torch_to_jax_array(done),
                truncated=torch_to_jax_array(truncated),
                extras={},
            )
            transitions.append(transition)
            obs = torch_to_jax_array(next_obs)

            if "final_info" in info:
                ep_returns = []
                for info in info["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={train_state.time_steps}, episode_return={info['episode']['r']}, episode_length={info['episode']['l']}"
                        )
                        ep_returns.append(info["episode"]["r"])

                wandb.log(
                    {"train/episode_return": np.mean(ep_returns)},
                    step=train_state.time_steps,
                )

        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            last_env_state=None,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )
        return transitions, train_state

    return collect_rollout


def make_eval_fn(
    env: ManagerBasedRlEnv, max_episode_steps: int, max_eval_episodes: int
) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Evaluate the policy in the environment
        key, eval_key = jax.random.split(key)

        # Reset the environment
        obs, _ = env.reset()
        obs = torch_to_jax_array(obs)
        episode_rewards = []
        running_rewards = torch.zeros(env.num_envs).to(env.device)

        for _ in range(max_episode_steps):
            # Select action
            action, _ = policy(eval_key, obs)
            # Step the environment
            next_obs, reward, done, truncated, info = env.step(
                jax_to_torch_array(action)
            )
            running_rewards += reward
            episode_rewards.append(running_rewards[done])
            running_rewards = torch.where(
                done, torch.zeros_like(running_rewards), running_rewards
            )
            obs = torch_to_jax_array(next_obs)

        episode_rewards = torch.cat(episode_rewards).cpu().numpy()

        return {
            "episode_return": np.mean(episode_rewards),
            "num_episodes": len(episode_rewards),
        }

    return evaluate
