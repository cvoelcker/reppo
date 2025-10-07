import time
import distrax
import logging
import gymnax
import jax
import jax.numpy as jnp
from mujoco_playground import State
import wandb

from gymnax.environments.environment import Environment
from flax import struct


def describe(values: jnp.ndarray, axis: tuple | int = 0) -> dict[str, jnp.ndarray]:
    """Compute basic statistics for a batch of values."""
    return {
        "mean": jnp.mean(values, axis=axis),
        "std": jnp.std(values, axis=axis),
        "min": jnp.min(values, axis=axis),
        "max": jnp.max(values, axis=axis),
    }


def merge_dicts(*prefix_dicts: tuple[str, dict], sep: str = "/") -> dict:
    """Merge metric dictionaries with a prefix for each key."""
    return {
        f"{prefix if prefix else ''}{sep if prefix else ''}{key}": value
        for prefix, metrics in prefix_dicts
        for key, value in metrics.items()
    }


def prefix_dict(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a prefix to all keys in a dictionary."""
    return {f"{prefix}{sep}{key}": value for key, value in metrics.items()}


def postfix_dict(postfix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a postfix to all keys in a dictionary."""
    return {f"{key}{sep}{postfix}": value for key, value in metrics.items()}


def filter_prefix(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Filter keys in a dictionary by a prefix."""
    return {
        key: value for key, value in metrics.items() if key.startswith(prefix + sep)
    }


def hl_gauss(inp, num_bins, vmin, vmax, epsilon=0.0):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(inp, vmin, max=vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(target_probs) / num_bins

    return (1 - epsilon) * target_probs + epsilon * uniform


@struct.dataclass
class MultiSampleLogProb:
    policy_action: jax.Array
    policy_action_log_prob: jax.Array
    action: jax.Array


def fast_multi_log_prob(
    key: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
    offset_scale: jax.Array,
) -> MultiSampleLogProb:
    """Computes 3 samples from a tanh squashed function
    - transformed loc and log_prob
    - sample with base scale
    - sample with scaled scale
    Args:
        key: JAX PRNG key.
        loc: Location of the distribution.
        scale: Scale parameter of the distribution.
        offset_scale: Offset scale for the distribution.
    """
    # log det factor

    # sample base gaussian noise with log prob
    base_noise, base_log_prob = distrax.Normal(
        jnp.zeros_like(loc), scale
    ).sample_and_log_prob(seed=key)
    base_log_prob = jnp.sum(base_log_prob, axis=-1)

    # sample with base scale
    base_sample = loc + base_noise
    base_sample_transformed = jnp.tanh(base_sample)
    # numerically stable jax tanh det jacobian https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    base_log_prob -= jnp.sum(
        2.0 * (jnp.log(2.0) - base_sample - jax.nn.softplus(-2.0 * base_sample)),
        axis=-1,
    )

    return MultiSampleLogProb(
        policy_action=base_sample_transformed,
        policy_action_log_prob=base_log_prob,
        action=jnp.tanh(loc + offset_scale * base_noise),
    )


def multi_softmax(x, dim=8, get_logits=False):
    inp_shape = x.shape
    if dim is not None:
        x = x.reshape(*x.shape[:-1], -1, dim)
    if get_logits:
        x = jax.nn.log_softmax(x, axis=-1)
    else:
        x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*inp_shape)


def multi_log_softmax(x, dim=8):
    if dim is not None:
        x = x.reshape(*x.shape[:-1], -1, dim)
        return jax.nn.log_softmax(x).reshape(x.shape)
    else:
        return jax.nn.log_softmax(x, axis=-1)


def simplical_softmax_cross_entropy(pred, target, dim=8):
    """Computes the cross-entropy loss for simplical softmax."""
    shape = pred.shape[-1]
    if dim is not None:
        pred = pred.reshape(*pred.shape[:-1], -1, dim)
        target = target.reshape(*target.shape[:-1], -1, dim)
    return jnp.sum(-target * jax.nn.log_softmax(pred, axis=-1), axis=-1).mean() / (
        shape / dim
    )


def make_log_callback():
    metric_history = []
    times = [time.perf_counter()]
    steps = [0]

    def log_callback(state, metrics):
        add_system_metrics(state, metrics)
        metrics = jax.tree.map(lambda x: jnp.array(x).mean().item(), metrics)
        steps.append(metrics["sys/time_step"])
        times.append(metrics["sys/time"])
        metric_history.append(metrics)

        # Use pop() with a default value of None in case 'advantages' key doesn't exist
        advantages = metrics.pop("train/advantages", None)
        print_logs(metrics)

        if advantages is not None:
            metrics["train/advantages"] = wandb.Histogram(advantages)
        wandb.log(metrics, step=int(metrics["sys/time_step"]))

    def add_system_metrics(state, metrics):
        metrics["sys/time_step"] = state.time_steps
        metrics["sys/time"] = time.perf_counter()
        num_env_steps = state.time_steps - steps[-1]
        seconds = metrics["sys/time"] - times[-1]
        sps = num_env_steps / seconds
        duration = metrics["sys/time"] - times[0]
        metrics["sys/wall_time"] = duration
        metrics["sys/sps"] = sps

    def print_logs(metrics):
        log_strs = []
        pev_category = None
        for k, v in sorted(metrics.items()):
            if isinstance(v, jnp.ndarray):
                metrics[k] = v.mean()
            elif isinstance(v, list):
                metrics[k] = jnp.array(v).mean()
            if "/" in k:
                category = k.split("/")[0]
                log_item = k.split("/")[1]
                if pev_category != category:
                    log_strs.append(f"  {category}/")
                    pev_category = category
                log_strs.append(f"    {log_item}={metrics[k]:.3f}")
            else:
                log_strs.append(f"  {k}={metrics[k]:.3f}")

        logging.info("\n" + "\n".join(log_strs))

    return log_callback


def init_env_state(
    key: jax.Array, env: Environment, num_envs: int
) -> tuple[jax.Array, gymnax.EnvState]:
    key, env_key = jax.random.split(key)
    env_key = jax.random.split(env_key, num_envs)
    obs, env_state = env.reset(env_key)
    if isinstance(env_state.unwrapped(), State):
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            env.episode_length,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)
    return obs, env_state
