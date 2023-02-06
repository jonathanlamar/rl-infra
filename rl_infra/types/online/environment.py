from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Protocol, TypeVar

from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass


# States will vary quite a bit between implementations, so I am just using this class as a type stub.
class State(SerializableDataClass):
    pass


Action = str
S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)


# This is really an interface, but I have to use ABC here because pydantic does not support mixing in with protocols.
class Transition(ABC, SerializableDataClass, Generic[S, A]):
    state: S
    action: A
    newState: S
    reward: float
    isTerminal: bool


class OnlineMetrics(ABC, SerializableDataClass):
    @abstractmethod
    def updateWithNewValues(self, other: Self) -> Self:
        ...


M = TypeVar("M", bound=OnlineMetrics)
T = TypeVar("T", bound=Transition)


class EpochRecord(ABC, SerializableDataClass, Generic[S, A, T, M]):
    epochNumber: int
    moves: list[T]

    @abstractmethod
    def computeOnlineMetrics(self) -> M:
        ...

    def append(self, transition: T) -> Self:
        return self.__class__(epochNumber=self.epochNumber, moves=self.moves + [transition])


class GameplayRecord(ABC, SerializableDataClass, Generic[S, A, T, M]):
    epochs: list[EpochRecord[S, A, T, M]]

    def computeOnlineMetrics(self) -> M:
        return reduce((lambda e1, e2: e1.updateWithNewValues(e2)), map(lambda e: e.computeOnlineMetrics(), self.epochs))

    def appendEpoch(self, epoch: EpochRecord[S, A, T, M]) -> Self:
        return self.__class__(epochs=self.epochs + [epoch])


class Environment(Protocol[S, A, T, M]):
    currentState: S
    currentEpochRecord: EpochRecord[S, A, T, M]
    currentGameplayRecord: GameplayRecord[S, A, T, M]

    def step(self, action: A) -> Transition[S, A]:
        ...

    def getReward(self, oldState: S, action: A, newState: S) -> float:
        ...

    def _updateEpoch(self, transition: T) -> None:
        self.currentEpochRecord = self.currentEpochRecord.append(transition)

    def _startNewEpoch(self, epochNumber: int | None = None) -> None:
        if epochNumber is None:
            epochNumber = self.currentEpochRecord.epochNumber + 1
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = self.currentEpochRecord.__class__(epochNumber=epochNumber, moves=[])
