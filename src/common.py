import typing
from typing import Optional, TypeVar

from gymnax import EnvState
import jax
from flax import struct
from flax.struct import PyTreeNode
from flax import nnx


Config = TypeVar("Config", bound=struct.PyTreeNode)
Key = jax.Array
Metrics = dict[str, jax.Array]


@struct.dataclass
class TrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array


class Transition(struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    truncated: jax.Array
    extras: dict[str, jax.Array]


class Policy(typing.Protocol):
    def __call__(
        self,
        key: Key,
        obs: PyTreeNode,
        state: Optional[PyTreeNode] = None,
    ) -> tuple[jax.Array, Optional[PyTreeNode]]:
        pass


class InitFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
    ) -> TrainState:
        pass


class LearnerFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        train_state: TrainState,
        batch: Transition,
    ) -> tuple[TrainState, Metrics]:
        pass


class RolloutFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        train_state: TrainState,
        policy: Policy,
    ) -> tuple[Transition, TrainState]:
        pass


class EvalFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        policy: Policy,
    ) -> dict[str, jax.Array]:
        pass


class PolicyFn(typing.Protocol):
    def __call__(
        self,
        train_state: TrainState,
        eval_mode: bool = False,
    ) -> Policy:
        pass


class LogCallback(typing.Protocol):
    def __call__(
        self,
        train_state: TrainState,
        metrics: dict[str, jax.Array],
    ) -> None:
        pass


class TrainFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        pass
