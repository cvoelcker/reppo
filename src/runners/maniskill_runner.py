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
                print(
                    f"global_step={train_state.time_steps}, episode_return={info['final_info']['episode']['return'].mean()}, success={info['final_info']['episode']['success_once'].mean()}"
                )
                
                
                wandb.log(
                    {
                        "train/episode_return": np.mean(info["final_info"]["episode"]["return"]),
                        "train/episode_sucess": np.mean(info["final_info"]["episode"]["success_once"])
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
