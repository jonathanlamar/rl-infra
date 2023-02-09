from typing import Protocol, TypeVar

from rl_infra.types.base_types import Metrics
from rl_infra.types.online.environment import EpochRecord, GameplayRecord
from rl_infra.types.online.transition import Action, State, Transition

A = TypeVar("A", bound=Action)
M = TypeVar("M", bound=Metrics)
S = TypeVar("S", bound=State)
T = TypeVar("T", bound=Transition)


class DataService(Protocol[S, A, T, M]):
    capacity: int

    def pushEpoch(self, epoch: EpochRecord[S, A, T, M]) -> None:
        ...

    def pushGameplay(self, gameplay: GameplayRecord[S, A, T, M]) -> None:
        ...

    def pushEntries(self, entries: list[T]) -> None:
        ...

    def sample(self, batchSize: int) -> list[T]:
        ...
