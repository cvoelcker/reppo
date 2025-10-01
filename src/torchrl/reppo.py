from dataclasses import dataclass, replace
import functools
import os
import random
import sys
import copy
import time

import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf

import wandb

from src.torchrl.reppo_util import EmpiricalNormalization, hl_gauss

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from tensordict import TensorDict
from torch.amp import GradScaler
from src.torchrl.envs import make_envs
from src.networks.torch_models import Actor, Critic


torch.set_float32_matmul_precision("medium")
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"


@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: Actor
    old_actor: Actor
    critic: Critic
    normalizer: EmpiricalNormalization
    critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    scaler: GradScaler

    def compile(self):
        self.actor.compile()
        self.old_actor.compile()
        self.critic.compile()
        self.normalizer.compile()
        self.critic_normalizer.compile()


def get_autocast_context(cfg: DictConfig):
    amp_enabled = (
        cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    )
    amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps"
        if cfg.platform.cuda and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if cfg.platform.amp_dtype == "bf16" else torch.float32
    return functools.partial(
        torch.amp.autocast,
        device_type=amp_device,
        dtype=amp_dtype,
        enabled=amp_enabled,
    )


def make_collect_fn(cfg: DictConfig, env):
    autocast = get_autocast_context(cfg)
    asymmetric_obs = env.asymmetric_obs

    def collect_fn(
        train_state: TrainState,
    ) -> tuple[TrainState, TensorDict, list[dict]]:
        transitions = []
        info_list = []
        obs = train_state.obs
        critic_obs = train_state.critic_obs

        for _ in range(cfg.hyperparameters.num_steps):
            with autocast():
                norm_obs = train_state.normalizer(obs)
                norm_critic_obs = train_state.critic_normalizer(critic_obs)
                with torch.inference_mode():
                    pi, _, _, _ = train_state.actor(norm_obs)
                    actions = pi.sample()

            next_obs, rewards, dones, truncations, infos = env.step(actions)

            if asymmetric_obs:
                next_critic_obs = infos["observations"]["critic"]
            else:
                next_critic_obs = next_obs

            with (
                torch.inference_mode(),
                autocast(),
            ):
                if (
                    cfg.env.get("has_final_obs", False)
                    and cfg.env.get("partial_reset", False)
                    and "final_observation" in infos
                ):
                    _next_obs = infos["final_observation"]
                    _next_critic_obs = _next_obs
                else:
                    _next_obs = next_obs
                    _next_critic_obs = next_critic_obs
                norm_next_obs = train_state.normalizer(_next_obs)
                next_pi, _, temperature, _ = train_state.actor(norm_next_obs)
                next_actions = next_pi.sample()
                next_log_probs = next_pi.log_prob(
                    next_actions.clip(-1 + 1e-6, 1 - 1e-6)
                ).sum(-1)
                norm_next_critic_obs = train_state.critic_normalizer(_next_critic_obs)
                next_value, _, _, next_embedding = train_state.critic(
                    norm_next_critic_obs, next_actions
                )
                rewards = (
                    rewards - cfg.hyperparameters.gamma * next_log_probs * temperature
                )

            transitions.append(
                TensorDict(
                    {
                        "observations": norm_obs,
                        "critic_observations": norm_critic_obs,
                        "actions": actions,
                        "log_probs": pi.log_prob(actions.clip(-0.999, 0.999)).sum(-1),
                        "rewards": rewards.unsqueeze(-1),
                        "next_embeddings": next_embedding,
                        "next_values": next_value.unsqueeze(-1),
                        "dones": dones.unsqueeze(-1).float(),
                        "truncations": truncations.unsqueeze(-1).float(),
                    },
                    batch_size=(env.num_envs,),
                )
            )
            info_list.append(infos)
            obs = next_obs
            critic_obs = next_critic_obs

        train_state = replace(train_state, obs=obs, critic_obs=critic_obs)
        return (
            train_state,
            torch.stack(transitions, dim=0),
            info_list,
        )

    return collect_fn


def make_postprocess_fn(cfg: DictConfig, env):
    @torch.compiler.disable()
    def compute_gve(rewards, dones, truncated, next_values, device: torch.device):
        gves = []
        last_gve = 0
        truncated[-1] = 1.0
        for t in reversed(range(cfg.hyperparameters.num_steps)):
            lambda_sum = (
                cfg.hyperparameters.lmbda * last_gve
                + (1.0 - cfg.hyperparameters.lmbda) * next_values[t]
            )
            delta = cfg.hyperparameters.gamma * torch.where(
                truncated[t].bool(), next_values[t], (1.0 - dones[t]) * lambda_sum
            )
            last_gve = rewards[t] + delta
            gves.insert(0, last_gve)
        return gves

    def postprocess(train_state: TrainState, transition: TensorDict):
        gve = compute_gve(
            rewards=transition["rewards"],
            dones=transition["dones"],
            truncated=transition["truncations"],
            next_values=transition["next_values"],
            device=train_state.device,
        )

        # Flatten all time and environment dimensions into a single batch dimension
        data = TensorDict(
            {
                "observations": transition["observations"],
                "critic_observations": transition["critic_observations"],
                "actions": transition["actions"],
                "rewards": transition["rewards"],
                "next_embeddings": transition["next_embeddings"],
                "next_values": transition["next_values"],
                "dones": transition["dones"],
                "truncations": transition["truncations"],
                "gve": torch.stack(gve),
            },
            batch_size=(
                cfg.hyperparameters.num_steps,
                cfg.hyperparameters.num_envs,
            ),
            device=train_state.device,
        )
        return data.float().flatten(0, 1).detach()

    return postprocess


def make_critic_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)

    def update(data: TensorDict):
        qnet = train_state.critic
        q_optimizer = train_state.critic_optimizer

        with autocast():
            critic_observations = data["critic_observations"]
            actions = data["actions"]
            targets = data["gve"]
            target_embeddings = data["next_embeddings"]
            truncations = data["truncations"].squeeze(-1)
            if cfg.env.get("partial_reset", False):
                truncation_mask = torch.ones_like(
                    truncations, dtype=torch.bool, device=train_state.device
                )
            else:
                truncation_mask = 1.0 - truncations
            qf_target_dist = hl_gauss(
                targets,
                cfg.hyperparameters.vmin,
                cfg.hyperparameters.vmax,
                cfg.hyperparameters.num_bins,
            )

            _, qf1, embedding, _ = qnet(critic_observations, actions)
            qf_loss = -(
                truncation_mask
                * torch.sum(qf_target_dist * F.log_softmax(qf1, dim=-1), dim=-1)
            ).mean()
            embedding_loss = (
                truncation_mask
                * F.mse_loss(
                    embedding,
                    target_embeddings,
                    reduction="none",
                ).mean(dim=-1)
            ).mean()

            qf_loss = qf_loss + cfg.hyperparameters.aux_loss_mult * embedding_loss

        q_optimizer.zero_grad(set_to_none=True)
        train_state.scaler.scale(qf_loss).backward()
        train_state.scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(), max_norm=cfg.hyperparameters.max_grad_norm
        )
        train_state.scaler.step(q_optimizer)
        train_state.scaler.update()
        logs_dict = {
            "critic_grad_norm": critic_grad_norm.detach(),
            "qf_loss": qf_loss.detach(),
            "qf_max": targets.max().detach(),
            "qf_min": targets.min().detach(),
            "qf_mean": targets.mean().detach(),
            "embedding_loss": embedding_loss.detach(),
        }
        return logs_dict

    return update


def make_actor_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)

    def update(data: TensorDict):
        actor = train_state.actor
        old_actor = train_state.old_actor
        qnet = train_state.critic
        actor_optimizer = train_state.actor_optimizer
        scaler = train_state.scaler
        critic_obs = data["critic_observations"]
        with autocast():
            pi, _, temperature, beta = actor(data["observations"])
            actions = pi.rsample()
            log_probs = pi.log_prob(actions.clip(-1 + 1e-6, 1 - 1e-6)).sum(-1)
            entropy = -log_probs
            qf, _, _, _ = qnet(critic_obs, actions)
            actor_loss = -qf + temperature.detach() * log_probs

            # compute KL
            old_pi, _, _, _ = old_actor(data["observations"])
            old_pi_actions = old_pi.sample((16,)).clip(-1 + 1e-6, 1 - 1e-6)
            old_log_probs = old_pi.log_prob(old_pi_actions).sum(-1).mean(0)
            new_pi_log_probs = pi.log_prob(old_pi_actions).sum(-1).mean(0)
            kl = old_log_probs - new_pi_log_probs

            if cfg.hyperparameters.actor_kl_clip_mode == "clipped":
                actor_loss = torch.where(
                    kl < cfg.hyperparameters.kl_bound,
                    actor_loss,
                    kl * beta.detach(),
                ).mean()
            elif cfg.hyperparameters.actor_kl_clip_mode == "full":
                actor_loss = actor_loss + kl * beta.detach()
            elif cfg.hyperparameters.actor_kl_clip_mode == "value":
                actor_loss = actor_loss
            else:
                raise ValueError(
                    f"Unknown actor kl clip mode: {cfg.hyperparameters.actor_kl_clip_mode}"
                )

            # temperature updates
            target_entropy = (
                actions.shape[-1] * cfg.hyperparameters.ent_target_mult
            )  # -0.5 * np.prod(envs.action_space.shape)
            entropy_loss = (target_entropy + entropy).detach().mean() * temperature

            lagrangian_loss = (
                -beta * (kl - cfg.hyperparameters.kl_bound).mean().detach()
            )

            actor_loss = (actor_loss + entropy_loss + lagrangian_loss).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), max_norm=cfg.hyperparameters.max_grad_norm
        )
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict = {
            "actor_grad_norm": actor_grad_norm.detach(),
            "actor_loss": actor_loss.detach(),
            "kl": kl.detach(),
            "entropy": entropy.detach(),
            "temperature": temperature.detach(),
            "lagrangian": beta.detach(),
            "entropy_loss": entropy_loss.detach(),
            "lagrangian_loss": lagrangian_loss.detach(),
        }
        return logs_dict

    return update


def make_evaluate_fn(cfg: DictConfig, eval_envs):
    autocast = get_autocast_context(cfg)

    @torch.inference_mode()
    def evaluate(
        train_state: TrainState, stochastic_eval: bool = False
    ) -> tuple[int | float | bool, int | float | bool]:
        train_state.normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=train_state.device)
        episode_lengths = torch.zeros(num_eval_envs, device=train_state.device)
        done_masks = torch.zeros(
            num_eval_envs, dtype=torch.bool, device=train_state.device
        )
        if cfg.env.type == "isaaclab" or cfg.env.asymmetric_obs:
            obs, _ = eval_envs.reset(random_start_init=False)
        else:
            obs, _ = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with autocast():
                obs = train_state.normalizer(obs)
                action_dist, det_actions, _, _ = train_state.actor(obs)
            if stochastic_eval:
                actions = action_dist.sample()
            else:
                actions = det_actions

            next_obs, rewards, dones, _, infos = eval_envs.step(actions)

            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        train_state.normalizer.train()

        if cfg.env.type == "maniskill":
            # combine log_infos
            info = {
                "info_return": infos["log_info"]["return"].mean(),
                "episode_len": infos["log_info"]["episode_len"].float().mean(),
                "success": infos["log_info"]["success"].float().mean(),
                "return": episode_returns.mean().item(),
            }
        else:
            info = {}

        return episode_returns.mean().item(), episode_lengths.mean().item(), info

    return evaluate


def configure_platform(cfg: DictConfig) -> DictConfig:
    cfg.platform.amp_enabled = (
        cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    )
    cfg.platform.amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps"
        if cfg.platform.cuda and torch.backends.mps.is_available()
        else "cpu"
    )
    return cfg


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="reppo",
)
def main(cfg):
    cfg = configure_platform(cfg)
    run_name = f"{cfg.env.name}_torch_{cfg.seed}"

    scaler = GradScaler(
        enabled=cfg.platform.amp_enabled and cfg.platform.amp_dtype == torch.float16
    )

    num_batches = cfg.hyperparameters.num_mini_batches
    batch_size = (
        cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps // num_batches
    )

    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        save_code=True,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.platform.torch_deterministic

    if not cfg.platform.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg.platform.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg.platform.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    envs, eval_envs = make_envs(cfg=cfg, device=device, seed=cfg.seed)

    n_act = envs.num_actions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if isinstance(envs.num_privileged_obs, int)
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs

    if cfg.hyperparameters.normalize_env:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        ent_start=cfg.hyperparameters.ent_start,
        kl_start=cfg.hyperparameters.kl_start,
        hidden_dim=cfg.hyperparameters.actor_hidden_dim,
        use_norm=cfg.hyperparameters.use_actor_norm,
        layers=cfg.hyperparameters.num_actor_layers,
        min_std=cfg.hyperparameters.actor_min_std,
        device=device,
    )
    old_actor = copy.deepcopy(actor)
    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=cfg.hyperparameters.num_bins,
        vmin=cfg.hyperparameters.vmin,
        vmax=cfg.hyperparameters.vmax,
        hidden_dim=cfg.hyperparameters.critic_hidden_dim,
        use_norm=cfg.hyperparameters.use_critic_norm,
        use_encoder_norm=False,
        encoder_layers=cfg.hyperparameters.num_critic_encoder_layers,
        head_layers=cfg.hyperparameters.num_critic_head_layers,
        pred_layers=cfg.hyperparameters.num_critic_pred_layers,
        device=device,
    )

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs, _ = envs.reset()
        critic_obs = obs

    train_state = TrainState(
        obs=obs,
        critic_obs=critic_obs,
        actor=actor,
        old_actor=old_actor,
        critic=qnet,
        normalizer=obs_normalizer,
        critic_normalizer=critic_obs_normalizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=q_optimizer,
        device=device,
        scaler=scaler,
    )

    # print(
    #     summary(
    #         train_state.critic,
    #         input_data=(critic_obs[:1], torch.zeros((1, n_act), device=device)),
    #         depth=10,
    #     )
    # )
    # print(summary(train_state.actor, input_data=(obs[:1],), depth=10))
    # create functions
    collect_fn = make_collect_fn(cfg, envs)
    postprocess_fn = make_postprocess_fn(cfg, envs)
    update_critic = make_critic_update_fn(cfg, train_state)
    update_actor = make_actor_update_fn(cfg, train_state)
    evaluate = make_evaluate_fn(cfg, eval_envs)

    if cfg.platform.compile:
        mode = "max-autotune-no-cudagraphs"
        update_critic = torch.compile(update_critic, mode=mode)
        update_actor = torch.compile(update_actor, mode=mode)
        postprocess_fn = torch.compile(postprocess_fn, mode=mode)
        train_state.compile()

    # TODO: Support checkpoint loading
    # if cfg.checkpoint_path:
    #     # Load checkpoint if specified
    #     torch_checkpoint = torch.load(
    #         f"{cfg.checkpoint_path}", map_location=device, weights_only=False
    #     )
    #     actor.load_state_dict(torch_checkpoint["actor_state_dict"])
    #     obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
    #     critic_obs_normalizer.load_state_dict(
    #         torch_checkpoint["critic_obs_normalizer_state"]
    #     )
    #     qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
    #     qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
    #     global_step = torch_checkpoint["global_step"]
    # else:
    global_step = 0
    total_env_steps = (
        cfg.hyperparameters.total_time_steps
        // (cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps)
        + 1
    )

    pbar = tqdm.tqdm(total=cfg.hyperparameters.total_time_steps, initial=global_step)
    start_time = None
    desc = ""

    eval_interval = total_env_steps // cfg.hyperparameters.num_eval
    stochastic_eval = cfg.env.get("stochastic_eval", False)

    while global_step < total_env_steps:
        if start_time is None and global_step >= cfg.measure_burnin:
            start_time = time.time()
            measure_burnin = global_step

        train_state, transition, infos = collect_fn(train_state)
        data = postprocess_fn(train_state, transition)

        for _ in range(cfg.hyperparameters.num_epochs):
            indices = torch.randperm(
                cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps,
                device=device,
            )
            data = data[indices].contiguous()
            for j in range(num_batches):
                mini_batch = data[j * batch_size : (j + 1) * batch_size]
                critic_logs_dict = update_critic(mini_batch)
                actor_logs_dict = update_actor(mini_batch)
                logs_dict = {
                    **critic_logs_dict,
                    **actor_logs_dict,
                }

        for param, target_param in zip(actor.parameters(), old_actor.parameters()):
            target_param.data.copy_(param.data)
        if start_time is not None:
            # @TODO: shouldn't that be env_steps per second?
            speed = (
                cfg.hyperparameters.num_envs
                * cfg.hyperparameters.num_steps
                * (global_step - measure_burnin)
                / (time.time() - start_time)
            )
            pbar.set_description(f"{speed: 4.4f} sps, " + desc)
            with torch.inference_mode():
                logs = {
                    "critic/qf_loss": logs_dict["qf_loss"].mean(),
                    "critic/qf_max": logs_dict["qf_max"].mean(),
                    "critic/qf_min": logs_dict["qf_min"].mean(),
                    "critic/qf_mean": logs_dict["qf_mean"].mean(),
                    "critic/embedding_loss": logs_dict["embedding_loss"].mean(),
                    "critic/critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                    "actor/actor_loss": logs_dict["actor_loss"].mean(),
                    "actor/actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                    "actor/kl": logs_dict["kl"].mean(),
                    "actor/entropy": logs_dict["entropy"].mean(),
                    "actor/temperature": logs_dict["temperature"].mean(),
                    "actor/lagrangian": logs_dict["lagrangian"].mean(),
                    "actor/entropy_loss": logs_dict["entropy_loss"].mean(),
                    "actor/lagrangian_loss": logs_dict["lagrangian_loss"].mean(),
                    "train/rewards_batch": data["rewards"].mean(),
                }

                if cfg.env.type == "maniskill":
                    logs.update(
                        {
                            "train/return": torch.stack(
                                [info["log_info"]["return"] for info in infos]
                            ).mean(),
                            "train/episode_len": torch.stack(
                                [info["log_info"]["episode_len"] for info in infos]
                            )
                            .float()
                            .mean(),
                            "train/success": torch.stack(
                                [info["log_info"]["success"] for info in infos]
                            )
                            .float()
                            .mean(),
                        }
                    )

                if eval_interval > 0 and global_step % eval_interval == 0:
                    print(f"Evaluating at global step {global_step}")
                    if stochastic_eval:
                        eval_avg_return, eval_avg_length, stoch_eval_info = evaluate(
                            train_state, stochastic_eval=stochastic_eval
                        )
                        eval_avg_return, eval_avg_length, eval_info = evaluate(
                            train_state
                        )
                        eval_info = {
                            **eval_info,
                            **{f"stoch/{k}": v for k, v in stoch_eval_info.items()},
                        }
                    else:
                        eval_avg_return, eval_avg_length, eval_info = evaluate(
                            train_state
                        )
                    if cfg.env.type in [
                        "humanoid_bench",
                        "isaaclab",
                        "mtbench",
                    ]:
                        # NOTE: Hacky way of evaluating performance, but just works
                        obs, _ = envs.reset()
                    logs["eval/avg_return"] = eval_avg_return
                    logs["eval/avg_length"] = eval_avg_length
                    for key, value in eval_info.items():
                        if isinstance(value, torch.Tensor):
                            logs[f"eval/{key}"] = value.mean().item()
                        elif isinstance(value, np.ndarray):
                            logs[f"eval/{key}"] = value.mean()
                        else:
                            logs[f"eval/{key}"] = value
                    print(
                        f"Eval return: {eval_avg_return:.2f}, length: {eval_avg_length:.2f}, env steps: {global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps} success rate: {eval_info.get('success', 0.0):.2f}"
                    )
            wandb.log(
                {
                    "speed": speed,
                    "frame": global_step
                    * cfg.hyperparameters.num_envs
                    * cfg.hyperparameters.num_steps,
                    **logs,
                },
                step=global_step
                * cfg.hyperparameters.num_envs
                * cfg.hyperparameters.num_steps,
            )

        global_step += 1
        pbar.update(n=cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps)


if __name__ == "__main__":
    main()
