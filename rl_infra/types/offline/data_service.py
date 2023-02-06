from abc import ABC
from typing import Generic, Protocol, TypeVar

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.online.environment import Action, Epoch, GameplayRecord, ModelOnlineMetrics, State, Transition

A = TypeVar("A", bound=Action)
DataDbRow = tuple[str, int, int]
DbEntry = TypeVar("DbEntry", bound=SerializableDataClass, covariant=False, contravariant=False)
M = TypeVar("M", bound=ModelOnlineMetrics)
S = TypeVar("S", bound=State)
T = TypeVar("T", bound=Transition)


class DataDbEntry(ABC, SerializableDataClass, Generic[T]):
    transition: T
    epoch: int
    move: int


class DataService(Protocol[DbEntry, S, A, T, M]):
    def pushEpoch(self, epoch: Epoch[S, A, T, M]) -> None:
        ...

    def pushGameplay(self, gameplay: GameplayRecord[S, A, T, M]) -> None:
        ...

    def pushEntries(self, entries: list[DbEntry]) -> None:
        ...

    def sample(self, batchSize: int) -> list[DbEntry]:
        ...
