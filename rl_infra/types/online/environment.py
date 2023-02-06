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


class ModelOnlineMetrics(ABC, SerializableDataClass):
    @abstractmethod
    def updateWithNewValues(self, other: Self) -> Self:
        ...


M = TypeVar("M", bound=ModelOnlineMetrics)
T = TypeVar("T", bound=Transition)


class Epoch(ABC, SerializableDataClass, Generic[S, A, T, M]):
    epochNumber: int
    moves: list[T]

    @abstractmethod
    def computeOnlineMetrics(self) -> M:
        ...


class GameplayRecord(ABC, SerializableDataClass, Generic[S, A, T, M]):
    epochs: list[Epoch[S, A, T, M]]

    def computeOnlineMetrics(self) -> M:
        return reduce((lambda e1, e2: e1.updateWithNewValues(e2)), map(lambda e: e.computeOnlineMetrics(), self.epochs))


class Environment(Protocol[S, A]):
    currentState: S

    def step(self, action: A) -> Transition[S, A]:
        ...

    def getReward(self, oldState: S, action: A, newState: S) -> float:
        ...
