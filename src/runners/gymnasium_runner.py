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
import wandb


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
            print(type(action))
            next_obs, reward, done, truncated, info = env.step(np.array(action))
            print(type(next_obs))
            # Record the transition
            transition = Transition(
                obs=jnp.array(obs),
                action=jnp.array(action),
                reward=jnp.array(reward),
                done=jnp.array(done),
                truncated=jnp.array(truncated),
                extras={},
            )
            transitions.append(transition)
            obs = next_obs

            if "final_info" in info:
                ep_returns = []
                for info in info["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={train_state.time_steps}, episode_return={info['episode']['r']}, episode_length={info['episode']['l']}"
                        )
                        ep_returns.append(info["episode"]["r"])

                wandb.log(
                    {
                        "train/episode_return": np.mean(
                            info["final_info"]["episode"]["return"]
                        ),
                        "train/episode_sucess": np.mean(
                            info["final_info"]["episode"]["success_once"]
                        ),
                    },
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
            dones.append(done)
            obs = next_obs
            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        num_episodes += 1
            if num_episodes >= max_eval_episodes:
                break

        return {
            "episode_return": np.mean(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": len(episode_rewards),
            "episode_lengths": episode_lengths,
        }

    return evaluate
