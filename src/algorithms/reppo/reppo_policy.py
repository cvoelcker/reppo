from typing import Callable
from dataclasses import dataclass

import jax
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space, Box
from src.algorithms.reppo.common import REPPOTrainState
from src.common import Policy

from src.normalization import Normalizer
import distrax


### Vanilla SAC style Policy

class REPPOPolicy(nnx.Module):
    def __init__(self, base: nnx.Module, normalizer: Normalizer | None, normalization_state, eval: bool, action_space: Space):
        self.base = base
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self._eval_mode = eval
        self.action_space = action_space

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        if self._eval_mode:
            action = self.base.det_action(x)
        else:
            pi = self.base(x, **kwargs)
            action = pi.sample(seed=key)
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}



@dataclass 
class LangevinConfig:
    # params for the step size (epsilon) computation
    a: float = 0.001
    b: float = 2.
    eps_const: bool = False # if we keep the epsilon the same across planning iterations

    grad_step_size: float = 1.0
    scale_noise_by_act_dim: bool = False

    # MPC params
    mpc_iterations: int = 3
    num_mpc_samples: int = 32
    num_policy_mpc_samples: int = 32
    mpc_temperature: float = 0.5

    gamma: float = 0.99

    exploration_noise_sigma: float = 0.3


class LangevinPolicy(nnx.Module):
    def __init__(self, actor: nnx.Module, critic: nnx.Module, normalizer: Normalizer | None, normalization_state, action_space: Space, config: LangevinConfig = LangevinConfig(), eval: bool = False):
        self.actor = actor
        self.critic = critic
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self.action_space = action_space
        self.config = config
        self._eval_mode = eval


    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        alpha = self.actor.temperature()

        if self._eval_mode:
            action = self.actor.det_action(x)
            if isinstance(self.action_space, Box):
                action = action.clip(-0.999, 0.999)
            return action, {}

        action = jax.vmap(get_langevin_action, in_axes=[None, None, 0, None, None, None])(
            self.actor, 
            self.critic, 
            x, 
            # eta_scaler, 
            alpha,
            key, 
            self.config
        )
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}

def get_grad_action(config: LangevinConfig, actor, critic, obs, actions, alpha):
    def get_values(x, a):
        # if not h.uniform_action_prior:
        #     act = jnp.tanh(act)
        vals = critic(x, a)["value"]
        return vals.mean()
    
    val_grad = jax.value_and_grad(get_values, argnums=1)
    prior_grad = jax.vmap(jax.grad(lambda a: actor.log_prob(a)))(actions)

    actions = jnp.clip(actions, -0.999, 0.999)
    
    act_val, act_grad = jax.vmap(
        val_grad, in_axes=[None, 0]
    )(
        obs, actions, 
    )

    return act_val, act_grad + prior_grad * alpha


def epsilon(config: LangevinConfig, it: int) -> float:
    return config.a / (config.b + it * (not config.eps_const))

    
def get_langevin_action(
        actor,
        critic,
        state: jax.Array, 
        # eta_scaler: jax.Array, 
        alpha: jax.Array,
        rand_key: jax.Array,
        config: LangevinConfig = LangevinConfig(), 
    ):
        rand_key, act_key = jax.random.split(rand_key)

        rand_key, act_key = jax.random.split(rand_key)
        pi = actor(state)
        actor_action = pi.sample(sample_shape=(config.num_policy_mpc_samples,), seed=act_key)

        mean = actor_action.mean(axis=0)
        std = jnp.ones_like(mean) * config.exploration_noise_sigma

        random_actions = mean + jax.random.normal(act_key, shape=[config.num_mpc_samples - config.num_policy_mpc_samples, *mean.shape]) * std

        actions = jnp.concatenate([actor_action, random_actions], axis=0)
        actions = jnp.clip(actions, -1 + 1e-4, 1 - 1e-4)

        values = critic(state[None].repeat(config.num_mpc_samples, axis=0), actions)["value"]
        top_softmax = jax.nn.softmax(values / config.mpc_temperature)
        top = jax.random.choice(act_key, jnp.arange(actions.shape[0]), p=top_softmax)


        for i in range(config.mpc_iterations):
            grad_key, eta_key, soft_key, rand_key = jax.random.split(rand_key, 4)

            values, grad = get_grad_action(config, pi, critic, state, actions, alpha)
            
            # eta = jnp.sqrt(epsilon(config, i)) * eta_scaler * jax.random.normal(eta_key, shape=actions.shape)
            eta = jnp.sqrt(epsilon(config, i)) * alpha * jax.random.normal(eta_key, shape=actions.shape)
            if config.scale_noise_by_act_dim:
                eta = eta / (actions.shape[-1] ** 0.5)

            action_delta = 0.5 * epsilon(config, i) * (grad * config.grad_step_size) + eta 
            # jax.debug.print("Langevin step {}/{}: eps = {}, max|grad| = {}, max|eta| = {}, max|delta| = {}", i+1, config.mpc_iterations, epsilon(config, i), jnp.abs(grad).max(), jnp.abs(eta).max(), jnp.abs(action_delta).max())
            actions = actions + action_delta
            actions = jnp.clip(actions, -1. + 1e-4, 1. - 1e-4)

            top_softmax = jax.nn.softmax(values.squeeze() / config.mpc_temperature)
            top = jax.random.choice(soft_key, jnp.arange(actions.shape[0]), p=top_softmax)

        action = actions[top]
                
        return action


def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = cfg.algorithm
    offset = None

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer()
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        if cfg.policy_method == "langevin":
            policy = LangevinPolicy(
                actor=actor_model,
                critic=critic_model,
                normalizer=normalizer if cfg.normalize_env else None,
                normalization_state=train_state.normalization_state,
                action_space=action_space,
                eval=eval,
            )
        else:
            policy = REPPOPolicy(
                base=actor_model,
                normalizer=normalizer if cfg.normalize_env else None,
                normalization_state=train_state.normalization_state,
                eval=eval,
                action_space=action_space,
            )
        policy.eval()


        return policy

    return policy_fn