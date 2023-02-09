from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, Protocol, Type, TypeVar

from pydantic import validator
from typing_extensions import Self

from rl_infra.types.base_types import Metrics, SerializableDataClass
from rl_infra.types.online.environment import EpochRecord, GameplayRecord
from rl_infra.types.online.transition import Action, State, Transition

A = TypeVar("A", bound=Action)
DbEntry = TypeVar("DbEntry", bound=SerializableDataClass, covariant=False, contravariant=False)
M = TypeVar("M", bound=Metrics)
S = TypeVar("S", bound=State)
T = TypeVar("T", bound=Transition)


class DataDbRow(NamedTuple):
    state: str
    action: str
    newState: str
    reward: int
    isTerminal: bool
    epoch: int
    move: int


class DataDbEntry(ABC, SerializableDataClass, Generic[S, A]):
    state: S
    action: A
    newState: S
    reward: int
    isTerminal: bool
    epoch: int
    move: int

    def toDbRow(self) -> DataDbRow:
        return DataDbRow(
            state=self.state.json(),
            action=self.action,
            newState=self.newState.json(),
            reward=self.reward,
            isTerminal=self.isTerminal,
            epoch=self.epoch,
            move=self.move,
        )

    @validator("state", "newState", pre=True)
    @classmethod
    @abstractmethod
    def _parseStateFromJson(cls: Type[Self], val: S | str) -> S:
        ...


class DataService(Protocol[DbEntry, S, A, T, M]):
    capacity: int

    def pushEpoch(self, epoch: EpochRecord[S, A, T, M]) -> None:
        ...

    def pushGameplay(self, gameplay: GameplayRecord[S, A, T, M]) -> None:
        ...

    def pushEntries(self, entries: list[DbEntry]) -> None:
        ...

    def sample(self, batchSize: int) -> list[DbEntry]:
        ...
