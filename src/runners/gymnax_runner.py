from gymnax.environments.environment import Environment
import jax
from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)


def make_eval_fn(env: Environment, max_episode_steps: int) -> EvalFn:
    def evaluation_fn(key: Key, policy: Policy):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            env_key = jax.random.split(env_key, env.num_envs)
            obs, env_state, reward, done, info = env.step(env_key, env_state, action)
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, env_state = env.reset(init_key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        return {
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            ),
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
        }

    return evaluation_fn


def make_rollout_fn(env: Environment, num_steps: int, num_envs: int) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment
        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, policy = carry

            # Select action
            key, act_key, step_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            step_key = jax.random.split(step_key, num_envs)
            next_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            # Record the transition
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                truncated=next_env_state.truncated,
                extras=info,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                policy,
            ), transition

        # Collect rollout via lax.scan taking steps in the environment
        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                policy,
            ),
            length=num_steps,
        )
        # Aggregate the transitions across all the environments to reset for the next iteration
        _, last_env_state, train_state, last_obs, _ = rollout_state

        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )

        return transitions, train_state

    return collect_rollout
