from typing import Callable

import distrax
import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.struct import PyTreeNode
from mujoco_playground import State, registry
from mujoco_playground._src.wrapper import wrap_for_brax_training
from omegaconf import DictConfig

import util
from src.normalization import Normalizer


class Transition(PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array
    truncated: jax.Array
    extras: dict


class REPPOTrainState(nnx.TrainState):
    actor: nnx.TrainState
    critic: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    normalization_state: dict | None
    last_env_state: State | None


def hl_gauss(values: jax.Array, num_bins: int, vmin: float, vmax: float, epsilon: float = 0.0) -> jax.Array:
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(values, vmin, max=vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32)
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*values.shape[:-1], num_bins)
    uniform = jnp.ones_like(target_probs) / num_bins
    return (1 - epsilon) * target_probs + epsilon * uniform


class Actor(nnx.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.network = nnx.Sequential(
            nnx.Linear(observation_size, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, action_size * 2, rngs=rngs),
        )
        self.log_lagrangian = nnx.Param(jnp.ones(1) * jnp.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * jnp.log(ent_start))

    def __call__(self, obs: jax.Array, deterministic: bool = False) -> distrax.Distribution | jax.Array:
        features = self.network(obs)
        mean, log_std = jnp.split(features, 2, axis=-1)
        std = jnp.exp(log_std) + 1e-6
        if deterministic:
            return jnp.tanh(mean)
        else:
            pi = distrax.Transformed(distrax.Normal(loc=mean, scale=std), distrax.Tanh())
            pi = distrax.Independent(pi, reinterpreted_batch_ndims=1)
            return pi

    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)


class Critic(nnx.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        *,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.network = nnx.Sequential(
            nnx.Linear(observation_size + action_size, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
        )
        self.feature_pred = nnx.Sequential(
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, 512 + 1, rngs=rngs),
        )
        self.q_head = nnx.Sequential(
            nnx.Linear(512, 512, rngs=rngs),
            nnx.LayerNorm(512, rngs=rngs),
            nnx.swish,
            nnx.Linear(512, num_bins, rngs=rngs),
        )
        self.zero_dist = nnx.Param(hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax))
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, obs: jax.Array, action: jax.Array) -> dict[str, jax.Array]:
        embed = self.network(jnp.concatenate([obs, action], axis=-1))
        pred_features = self.feature_pred(embed)
        logits = self.q_head(embed) + self.zero_dist.value * 40.0
        probs = jax.nn.softmax(logits, axis=-1)
        value = probs.dot(jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True))
        return {
            "value": value,
            "logits": logits,
            "probs": probs,
            "embed": embed,
            "pred_features": pred_features[..., :-1],
            "pred_rew": pred_features[..., -1:],
        }


def make_train_fn(cfg: DictConfig) -> Callable[[jax.random.PRNGKey], tuple[REPPOTrainState, dict[str, jax.Array]]]:
    normalizer = Normalizer()
    env = registry.load(cfg.env.name)
    env = wrap_for_brax_training(env)

    def init(key: jax.random.PRNGKey):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)
        norm_state = normalizer.init(
            jax.tree.map(
                lambda x: jnp.zeros_like(x, dtype=float),  # type: ignore
                env.observation_size,
            )
        )
        actor = Actor(
            observation_size=env.observation_size,
            action_size=env.action_size,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            rngs=rngs,
        )
        critic = Critic(
            observation_size=env.observation_size,
            action_size=env.action_size,
            num_bins=cfg.num_bins,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            rngs=rngs,
        )
        if cfg.anneal_lr:
            lr = optax.linear_schedule(
                init_value=cfg.learning_rate, end_value=0.0, transition_steps=cfg.total_time_steps
            )
        else:
            lr = cfg.learning_rate
        opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr))

        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params=nnx.state(actor),
            tx=optax.set_to_zero(),
            actor=nnx.TrainState.create(graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=opt),
            critic=nnx.TrainState.create(graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=opt),
            actor_target=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor),
                params=nnx.state(actor),
                tx=optax.set_to_zero(),
            ),
            iteration=0,
            time_steps=0,
            normalization_state=norm_state,
            last_env_state=None,
        )

    def policy(
        key: jax.random.PRNGKey, obs: jax.Array, train_state: REPPOTrainState, eval: bool = False
    ) -> tuple[jax.Array, dict]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        obs = normalizer.normalize(train_state.normalization_state, obs)
        if eval:
            action: jax.Array = actor_model(obs, deterministic=True)
        else:
            pi = actor_model(obs, deterministic=False)
            action = pi.sample(seed=key)

        return action

    def collect_rollout(key: jax.random.PRNGKey, train_state: REPPOTrainState) -> tuple[Transition, REPPOTrainState]:
        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state = carry
            # Select action
            key, act_key = jax.random.split(key)
            action = policy(act_key, env_state.obs, train_state)
            # Take a step in the environment
            action = jnp.clip(action, -0.999, 0.999)
            next_env_state = env.step(env_state, action)
            # Record the transition
            transition = Transition(
                obs=env_state.obs,
                action=action,
                reward=next_env_state.reward,
                next_obs=next_env_state.info["raw_obs"],
                done=next_env_state.done,
                truncated=next_env_state.info["truncation"],
                extras=next_env_state.info,  # holds true next state in case of truncation
            )
            return (
                key,
                next_env_state,
            ), transition

        # Collect rollout via lax.scan taking steps in the environment
        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
            ),
            length=cfg.num_steps,
        )
        # Aggregate the transitions across all the environments to reset for the next iteration
        _, last_env_state = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )
        return transitions, train_state

    def evaluate(key: jax.random.PRNGKey, train_state: REPPOTrainState) -> dict[str, jax.Array]:

        def step_env(carry, _):
            key, env_state = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action = policy(act_key, env_state.obs, train_state, eval=True)
            env_key = jax.random.split(env_key, cfg.num_envs)
            action = jnp.clip(action, -0.999, 0.999)
            next_env_state = env.step(env_state, action)
            return (key, next_env_state), next_env_state.info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, cfg.num_envs)
        env_state = env.reset(init_key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state),
            xs=None,
            length=cfg.env.episode_length,
        )
        episodic_metrics = jax.tree.map(lambda x: jnp.mean(x, where=infos["episode_done"]), infos["episode_metrics"])
        return episodic_metrics

    def critic_loss(params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_output = critic_model(minibatch.obs, minibatch.action)
        target_values = minibatch.extras["target_values"]
        target_cat = jax.vmap(hl_gauss, in_axes=(0, None, None, None))(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
        critic_pred = critic_output["logits"]
        critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)

        # Aux loss
        pred = critic_output["pred_features"]
        pred_rew = critic_output["pred_rew"]
        value = critic_output["value"]
        aux_loss = optax.squared_error(pred, minibatch.extras["next_emb"])
        aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1)) * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
        )
        # compute l2 error for logging
        critic_loss = optax.squared_error(
            value,
            target_values,
        )
        critic_loss = jnp.mean(critic_loss)
        loss = jnp.mean((1.0 - minibatch.truncated) * (critic_update_loss + cfg.aux_loss_coeff * aux_loss))
        return loss, dict(
            value_mse=critic_loss.mean(),
            cross_entropy_loss=critic_update_loss.mean(),
            loss=loss.mean(),
            aux_loss=aux_loss.mean(),
            rew_aux_loss=aux_rew_loss.mean(),
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
        )

    def actor_loss(params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition):
        critic_target_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_model = nnx.merge(train_state.actor.graphdef, params)
        actor_target_model = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
        pi = actor_model(minibatch.obs)
        old_pi = actor_target_model(minibatch.obs)

        # policy KL constraint
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        alpha = jax.lax.stop_gradient(actor_model.temperature())
        pred_action, log_prob = pi.sample_and_log_prob(seed=minibatch.extras["action_key"])
        obs = minibatch.obs
        critic_pred = critic_target_model(obs, pred_action)
        value = critic_pred["value"]
        actor_loss = log_prob * alpha - value
        entropy = -log_prob
        action_size_target = env.action_size * cfg.ent_target_mult
        lagrangian = actor_model.lagrangian()
        loss = jnp.mean(
            jnp.where(
                kl < cfg.kl_bound,
                actor_loss,
                kl * jax.lax.stop_gradient(lagrangian),
            )
        )

        # SAC target entropy loss
        target_entropy = action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(target_entropy)

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - cfg.kl_bound)

        # total loss
        loss += jnp.mean(target_entropy_loss)
        loss += jnp.mean(lagrangian_loss)
        return loss, dict(
            actor_loss=actor_loss.mean(),
            loss=loss.mean(),
            temp=actor_model.temperature(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean(),
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss.mean(),
            entropy=entropy.mean(),
            entropy_loss=target_entropy_loss.mean(),
            target_values=minibatch.extras["target_values"].mean(),
        )

    def compute_policy_kl(minibatch: Transition, pi: distrax.Distribution, old_pi: distrax.Distribution) -> jax.Array:
        old_pi_action, old_pi_act_log_prob = old_pi.sample_and_log_prob(
            sample_shape=(4,), seed=minibatch.extras["kl_key"]
        )
        old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)
        old_pi_act_log_prob = old_pi_act_log_prob.mean(0)
        pi_act_log_prob = pi.log_prob(old_pi_action).mean(0)
        kl = old_pi_act_log_prob - pi_act_log_prob
        return kl

    def update(train_state: REPPOTrainState, batch: Transition):
        # Sample data at indices from the batch
        critic_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)
        output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
        critic_train_state = train_state.critic.apply_gradients(grads)
        train_state = train_state.replace(
            critic=critic_train_state,
        )
        critic_metrics = output[1]
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
        grad_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
        grad_norm = jax.tree.reduce(lambda x, y: x + y, grad_norm)

        grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads)
        actor_train_state = train_state.actor.apply_gradients(grads)
        train_state = train_state.replace(
            actor=actor_train_state,
        )
        actor_metrics = output[1]
        return train_state, {
            **util.prefix_dict("critic", critic_metrics),
            **util.prefix_dict("actor", actor_metrics),
            "grad_norm": grad_norm,
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
        indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
        minibatch_idxs = indices.reshape((cfg.num_mini_batches, mini_batch_size))
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)
        minibatches.extras["action_key"] = jax.random.split(act_key, cfg.num_mini_batches)
        minibatches.extras["kl_key"] = jax.random.split(kl_key, cfg.num_mini_batches)

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        metrics_mean = jax.tree.map(lambda x: x.mean(0), metrics)
        return (
            train_state,
            metrics_mean,
        )

    def nstep_lambda(batch: Transition):
        def loop(returns: jax.Array, transition: Transition):
            done = transition.done
            reward = transition.extras["soft_reward"]
            next_value = transition.extras["next_value"]
            truncated = transition.truncated
            lambda_sum = cfg.lmbda * returns + (1 - cfg.lmbda) * next_value
            lambda_return = reward + cfg.gamma * jnp.where(truncated, next_value, (1.0 - done) * lambda_sum)
            return lambda_return, lambda_return

        _, lambda_return = jax.lax.scan(
            f=loop,
            init=batch.extras["next_value"][-1],
            xs=batch,
            reverse=True,
        )
        return lambda_return

    def compute_extras(key: jax.random.PRNGKey, train_state: REPPOTrainState, batch: Transition):
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        key, next_act_key = jax.random.split(key)
        next_pi = actor_model(batch.next_obs, deterministic=False)
        next_action = next_pi.sample(seed=next_act_key)
        true_next_action = jnp.where(
            batch.truncated[..., None],
            next_action,
            jnp.concatenate([batch.action[1:], next_action[-1:]], axis=0),
        )
        true_next_action = jnp.clip(true_next_action, -0.999, 0.999)
        true_next_log_prob = next_pi.log_prob(true_next_action)
        soft_reward = batch.reward - cfg.gamma * true_next_log_prob * actor_model.temperature()

        key, next_act_key = jax.random.split(key)
        next_sample_actions = next_pi.sample(seed=next_act_key, sample_shape=(cfg.num_action_samples,))
        next_critic_output = critic_model(
            jnp.repeat(batch.next_obs[None, ...], cfg.num_action_samples, axis=0),
            next_sample_actions,
        )
        next_emb = next_critic_output["embed"][0]
        next_values = next_critic_output["value"].mean(0)

        # compute log prob for each action.
        pi = actor_model(batch.obs, deterministic=False)
        extras = {
            "soft_reward": soft_reward,
            "next_value": next_values,
            "next_emb": next_emb,
            "log_prob": pi.log_prob(batch.action),
        }
        return extras

    def learner_fn(
        key: jax.random.PRNGKey, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        new_norm_state = normalizer.update(train_state.normalization_state, batch.obs)
        batch = batch.replace(
            obs=normalizer.normalize(train_state.normalization_state, batch.obs),
            next_obs=normalizer.normalize(train_state.normalization_state, batch.next_obs),
        )
        train_state = train_state.replace(normalization_state=new_norm_state)

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)
        batch.extras["target_values"] = nstep_lambda(batch=batch)

        # Reshape data to (num_steps * num_envs, ...)
        batch = jax.tree.map(lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), batch)
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(params=train_state.actor.params),
        )
        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch),
            init=train_state,
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    def train_step(state: REPPOTrainState, key: jax.random.PRNGKey) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        key, rollout_key, learn_key = jax.random.split(key, 3)
        # Collect trajectories from `state`
        transitions, state = collect_rollout(key=rollout_key, train_state=state)
        # Execute an update to the policy with `transitions`
        state, update_metrics = learner_fn(key=learn_key, train_state=state, batch=transitions)
        metrics = {**update_metrics, **update_metrics}
        state = state.replace(iteration=state.iteration + 1)
        return state, metrics

    def train_eval_step(key, train_state):
        train_key, eval_key = jax.random.split(key)
        eval_interval = int((cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval)
        eval_interval = max(eval_interval, 1)
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        eval_metrics = evaluate(eval_key, train_state)
        metrics = {
            **util.prefix_dict("train", train_metrics),
            **util.prefix_dict("eval", eval_metrics),
        }

        return train_state, metrics

    def train_eval_loop_body(train_state: REPPOTrainState, key: jax.random.PRNGKey) -> tuple[REPPOTrainState, dict]:
        # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(train_eval_step)(jax.random.split(subkey, cfg.num_seeds), train_state)
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        metrics["iteration"] = train_state.iteration[0]
        metrics["time_steps"] = train_state.time_steps[0]
        jax.debug.callback(util.log_callback, metrics)
        return train_state, metrics

    def init_train_state(key: jax.random.PRNGKey) -> REPPOTrainState:
        key, env_key = jax.random.split(key)
        train_state = init(key)
        env_state = env.reset(jax.random.split(env_key, cfg.num_envs))
        key, randomize_steps_key = jax.random.split(key)
        env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            env_state.info["steps"].shape,
            0,
            env.episode_length,
        ).astype(jnp.float32)
        train_state = train_state.replace(last_env_state=env_state)
        return train_state

    def scan_train_fn(key: jax.random.PRNGKey) -> tuple[REPPOTrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(init_train_state)(jax.random.split(init_key, cfg.num_seeds))
        keys = jax.random.split(key, cfg.num_eval)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=train_eval_loop_body, init=train_state, xs=keys)
        return state, metrics

    return jax.jit(scan_train_fn) if cfg.jit else scan_train_fn


@hydra.main(version_base=None, config_path="../config", config_name="reppo_mj_playground")
def main(cfg: dict) -> None:
    train_fn = make_train_fn(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    final_state, metrics = train_fn(key)


if __name__ == "__main__":
    main()
