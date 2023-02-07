from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Protocol, TypeVar

from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.online.transition import Action, State, Transition


class OnlineMetrics(ABC, SerializableDataClass):
    @abstractmethod
    def updateWithNewValues(self, other: Self) -> Self:
        ...


S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)
T = TypeVar("T", bound=Transition)
M = TypeVar("M", bound=OnlineMetrics)


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

    def startNewEpoch(self) -> None:
        ...
