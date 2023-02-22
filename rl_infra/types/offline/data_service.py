from typing import Protocol, TypeVar, Iterable

from rl_infra.types.base_types import Metrics
from rl_infra.types.online.environment import EpochRecord, GameplayRecord
from rl_infra.types.online.transition import Action, State, Transition

A = TypeVar("A", bound=Action)
M = TypeVar("M", bound=Metrics, contravariant=True)
S = TypeVar("S", bound=State)


class DataService(Protocol[S, A, M]):
    capacity: int

    def pushEpoch(self, epoch: EpochRecord[S, A, M]) -> None:
        ...

    def pushGameplay(self, gameplay: GameplayRecord[S, A, M]) -> None:
        ...

    def pushEntries(self, entries: Iterable[Transition[S, A]]) -> None:
        ...

    def sample(self, batchSize: int) -> Iterable[Transition[S, A]]:
        ...

    def keepNewRowsDeleteOld(self, sgn: int, numToKeep: int) -> None:
        ...
