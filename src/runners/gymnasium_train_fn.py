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

    if rollout_fn is None:
        rollout_fn = make_gymnasium_rollout_fn(env, num_steps, num_envs)

    if eval_fn is None:
        eval_fn = make_gymnasium_eval_fn(eval_env, max_episode_steps)

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_eval
        train_steps_per_iteration = num_train_steps // num_iterations
        eval_interval = num_train_steps // num_iterations
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(
            last_obs=jax.tree.map(jnp.array, obs), last_env_state=None
        )
        for i in range(num_iterations):
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key = jax.random.split(key, 3)
                # Collect trajectories from `state`
                policy = policy_fn(state, False)
                transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )
                # Execute an update to the policy with `transitions`
                state, train_metrics = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

            key, eval_key = jax.random.split(key)
            eval_metrics = eval_fn(eval_key, policy)
            metrics = {
                "time_step": state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
            }
            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, metrics)
        return state, metrics

    return loop_train_fn


def make_gymnasium_rollout_fn(
    env: gymnasium.Env, num_steps: int, num_envs: int
) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment

        transitions = []
        obs = train_state.last_obs
        for _ in range(num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(
                np.array(action)
            )
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

        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            last_env_state=None,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )
        return transitions, train_state

    return collect_rollout


def make_gymnasium_eval_fn(env: gymnasium.Env, max_episode_steps: int) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Evaluate the policy in the environment
        key, eval_key = jax.random.split(key)

        # Reset the environment
        obs, _ = env.reset()
        done = False
        dones = []
        episode_rewards = []
        episode_lengths = []
        for _ in range(max_episode_steps + 1):
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

        return {
            "episode_return": np.mean(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": len(episode_rewards),
            "episode_lengths": episode_lengths,
        }

    return evaluate
