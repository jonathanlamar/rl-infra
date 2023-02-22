from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Protocol, TypeVar

from typing_extensions import Self

from rl_infra.types.base_types import Metrics, SerializableDataClass
from rl_infra.types.online.transition import Action, State, Transition

S_co = TypeVar("S_co", bound=State, covariant=True)
A_co = TypeVar("A_co", bound=Action, covariant=True)
M_co = TypeVar("M_co", bound=Metrics, covariant=True)


class EpochRecord(ABC, SerializableDataClass, Generic[S_co, A_co, M_co]):
    moves: list[Transition[S_co, A_co]]

    @abstractmethod
    def computeOnlineMetrics(self) -> M_co:
        ...

    def append(self, transition: Transition[S_co, A_co]) -> Self:
        return self.__class__(moves=self.moves + [transition])


class GameplayRecord(ABC, SerializableDataClass, Generic[S_co, A_co, M_co]):
    epochs: list[EpochRecord[S_co, A_co, M_co]]

    def computeOnlineMetrics(self) -> M_co:
        return reduce((lambda e1, e2: e1.updateWithNewValues(e2)), map(lambda e: e.computeOnlineMetrics(), self.epochs))

    def appendEpoch(self, epoch: EpochRecord[S_co, A_co, M_co]) -> Self:
        return self.__class__(epochs=self.epochs + [epoch])


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)
M = TypeVar("M", bound=Metrics)


class Environment(Protocol[S, A, M]):
    currentState: S
    currentEpochRecord: EpochRecord[S, A, M]
    currentGameplayRecord: GameplayRecord[S, A, M]

    def step(self, action: A) -> Transition[S, A]:
        ...

    def getReward(self, oldState: S, action: A, newState: S) -> float:
        ...

    def startNewEpoch(self) -> None:
        ...
