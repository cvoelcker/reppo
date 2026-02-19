import functools
from collections import defaultdict
from typing import Callable

import distrax
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import mani_skill.envs
import optax
import torch
from flax import nnx
from flax.struct import PyTreeNode
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig

import util
from src.normalization import Normalizer
from src.util import prefix_dict


def make_env(cfg: DictConfig, eval: bool = False):
    env_kwargs = cfg.env.kwargs if "kwargs" in cfg.env else {}
    if cfg.env.control_mode is not None:
        env_kwargs["control_mode"] = cfg.env.control_mode
    reconfiguration_freq = cfg.env.eval_reconfiguration_freq if eval else cfg.env.reconfiguration_freq
    partial_resets = cfg.env.eval_partial_reset if eval else cfg.env.partial_reset
    envs = gym.make(
        cfg.env.name,
        num_envs=cfg.num_envs,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    if cfg.env.capture_video:
        if cfg.env.save_train_video_freq is not None or eval:
            video_dir = "train_videos" if not eval else "eval_videos"
            save_video_trigger = lambda x: (x // cfg.num_steps) % cfg.env.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=video_dir,
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=cfg.num_steps,
                video_fps=30,
            )
    envs = ManiSkillVectorEnv(
        envs,
        cfg.num_envs,
        ignore_terminations=not partial_resets,
        record_metrics=True,
    )

    return envs


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
    last_obs: PyTreeNode | None


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


def to_jax(x: torch.Tensor) -> jax.Array:
    return jax.tree.map(lambda y: jax.dlpack.from_dlpack(y.contiguous()), x)


def to_torch(x: jax.Array) -> torch.Tensor:
    return jax.tree.map(lambda y: torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(y)), x)


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
    env = make_env(cfg)
    eval_env = make_env(cfg, eval=True)

    def init(key: jax.random.PRNGKey):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)
        norm_state = normalizer.init(
            jax.tree.map(
                lambda x: jnp.zeros_like(x, dtype=float),
                env.single_observation_space.sample(),
            )
        )
        actor = Actor(
            observation_size=env.single_observation_space.shape[0],
            action_size=env.single_action_space.shape[0],
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            rngs=rngs,
        )
        critic = Critic(
            observation_size=env.single_observation_space.shape[0],
            action_size=env.single_action_space.shape[0],
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
            last_obs=None,
        )

    @functools.partial(jax.jit, static_argnums=3)
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

        return action, {}

    def collect_rollout(key: jax.random.PRNGKey, train_state: REPPOTrainState) -> tuple[Transition, REPPOTrainState]:
        transitions = []
        obs = train_state.last_obs
        metrics = defaultdict(list)
        for _ in range(cfg.num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(key=act_key, obs=obs, train_state=train_state, eval=False)
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(to_torch(action))
            if cfg.env.partial_reset:
                # maniskill continues bootstrap on terminated, which playground does on truncated.
                # This unifies the interfaces in a very hacky way
                done = torch.zeros_like(terminated, dtype=torch.bool)
                truncated = torch.logical_or(terminated, truncated)
            else:
                done = torch.logical_or(terminated, truncated)
                truncated = torch.zeros_like(done, dtype=torch.bool)

            # Record the transition
            if "final_observation" in info:
                mask = info["_final_info"]
                true_next_obs = jax.tree.map(
                    lambda x, y: torch.where(mask.unsqueeze(1), x, y),
                    info["final_observation"],
                    next_obs,
                )
                final_info = info["final_info"]
                for k, v in final_info["episode"].items():
                    metrics[k].append(to_jax(v[mask].float().mean()))
            else:
                true_next_obs = next_obs

            transition = Transition(
                obs=obs,
                action=action,
                reward=to_jax(reward),
                next_obs=to_jax(true_next_obs),
                done=to_jax(done),
                truncated=to_jax(truncated),
                extras={},
            )
            transitions.append(transition)
            obs = to_jax(next_obs)

        metrics = {k: jnp.stack(v).mean() for k, v in metrics.items()}
        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )
        return transitions, train_state, metrics

    def evaluate(key: jax.random.PRNGKey, train_state: REPPOTrainState) -> dict[str, jax.Array]:
        obs, _ = eval_env.reset()
        metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(cfg.env.max_episode_steps):
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, to_jax(obs), train_state, eval=True)
            next_obs, reward, terminated, truncated, infos = eval_env.step(to_torch(action))
            if "final_info" in infos:
                mask = infos["_final_info"]
                num_episodes += mask.sum()
                for k, v in infos["final_info"]["episode"].items():
                    metrics[k].append(v[mask])
            obs = next_obs

        eval_metrics = {}
        metrics = to_jax(metrics)
        for k, v in metrics.items():
            eval_metrics[f"{k}_std"] = jnp.std(jnp.stack(v))
            eval_metrics[k] = jnp.mean(jnp.stack(v))
        return eval_metrics

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
        action_size_target = env.action_space.shape[0] * cfg.ent_target_mult
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

    @jax.jit
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

    def train_fn(key: jax.random.PRNGKey) -> tuple[REPPOTrainState, dict]:

        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        train_steps_per_eval = max(num_train_steps // cfg.num_eval, 1)
        key, init_key = jax.random.split(key)
        state = init(init_key)
        obs, _ = env.reset()
        state = state.replace(last_obs=to_jax(obs))
        for i in range(cfg.num_eval):
            for _ in range(train_steps_per_eval):
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state, env_info = collect_rollout(key=rollout_key, train_state=state)
                state, train_metrics = learner_fn(key=learn_key, train_state=state, batch=transitions)
                state = state.replace(iteration=state.iteration + 1)
            key, eval_key = jax.random.split(key)
            eval_metrics = evaluate(eval_key, state)
            metrics = {
                "iteration": state.iteration,
                "time_steps": state.time_steps,
                **prefix_dict("train", train_metrics),
                **prefix_dict("eval", eval_metrics),
                **prefix_dict("env", env_info),
            }
            util.log_callback(metrics)
        return state, metrics

    return train_fn


@hydra.main(version_base=None, config_path="../config", config_name="reppo_maniskill")
def main(cfg: dict) -> None:
    train_fn = make_train_fn(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    final_state, metrics = train_fn(key)


if __name__ == "__main__":
    main()
