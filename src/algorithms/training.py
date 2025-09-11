import logging
import time
import gymnasium
from gymnax.environments.environment import Environment
import jax
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
    TrainFn,
    TrainState,
    Transition,
)
from src.algorithms import utils
import jax.numpy as jnp


def make_scan_train_fn(
    env: Environment | tuple[Environment, Environment],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    num_seeds: int,
    max_episode_steps: int,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    eval_fn: EvalFn | None = None,
    rollout_fn: RolloutFn | None = None,
    log_callback: LogCallback | None = None,
) -> TrainFn:
    from src.runners.gymnax_runner import (
        make_eval_fn as make_gymnax_eval_fn,
        make_rollout_fn as make_gymnax_rollout_fn,
    )

    # Initialize the environment and wrap it to admit vectorized behavior.
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    eval_interval = int((total_time_steps / (num_steps * num_envs)) // num_eval)

    if eval_fn is None:
        eval_fn = make_gymnax_eval_fn(eval_env, max_episode_steps)

    if rollout_fn is None:
        rollout_fn = make_gymnax_rollout_fn(env, num_steps=num_steps, num_envs=num_envs)

    if log_callback is None:
        log_callback = lambda state, metrics: None

    def train_step(
        state: TrainState, key: Key
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        key, rollout_key, learn_key = jax.random.split(key, 3)
        # Collect trajectories from `state`
        policy = policy_fn(state, False)
        transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )
        # Execute an update to the policy with `transitions`
        state, update_metrics = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        metrics = {**update_metrics, **update_metrics}
        state = state.replace(iteration=state.iteration + 1)
        return state, metrics

    def train_eval_step(key, train_state):
        train_key, eval_key = jax.random.split(key)
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        policy = policy_fn(train_state, True)
        eval_metrics = eval_fn(eval_key, policy)
        metrics = {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

        return train_state, metrics

    def train_eval_loop_body(
        train_state: TrainState, key: Key
    ) -> tuple[TrainState, dict]:
        # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(train_eval_step)(
            jax.random.split(subkey, num_seeds), train_state
        )
        jax.debug.callback(log_callback, train_state, metrics)
        return train_state, metrics

    def init_train_state(key: Key) -> TrainState:
        key, env_key = jax.random.split(key)
        train_state = init_fn(key)
        obs, env_state = utils.init_env_state(key=env_key, env=env, num_envs=num_envs)
        train_state = train_state.replace(last_obs=obs, last_env_state=env_state)
        return train_state

    # Define the training loop
    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(init_train_state)(jax.random.split(init_key, num_seeds))
        keys = jax.random.split(key, num_iterations)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=train_eval_loop_body, init=train_state, xs=keys)
        return state, metrics

    return jax.jit(scan_train_fn)


def make_loop_train_fn(
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    train_log_interval: int,
    max_episode_steps: int,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
):
    from src.runners.gymnasium_runner import (
        make_eval_fn as make_gymnasium_eval_fn,
        make_rollout_fn as make_gymnasium_rollout_fn,
    )

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
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(
            last_obs=jax.tree.map(jnp.array, obs), last_env_state=None
        )
        logging.info(f"Starting training for {num_iterations} iterations.")
        logging.info(f"Train steps per iteration: {train_steps_per_iteration}.")
        logging.info(f"Total time steps: {total_time_steps}.")
        
      
        step = 0
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

                if step % train_log_interval == 0:
                    log_callback(state, utils.prefix_dict("train", train_metrics))
                step += 1
            policy = policy_fn(state, True)
            key, eval_key = jax.random.split(key)
            eval_metrics = eval_fn(eval_key, policy)
            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, utils.prefix_dict("eval", eval_metrics))
        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn
