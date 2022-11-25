from dataclasses import dataclass
from typing import Protocol, TypeVar


# States will vary quite a bit between implementations, so I am just using this
# class as a type stub.
class State:
    pass


Action = str
S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)


@dataclass
class Transition(Protocol[S, A]):
    state: S
    action: A
    newState: S
    reward: float
    isTerminal: bool


class Environment(Protocol[S, A]):
    currentState: S

    def step(self, action: A) -> Transition[S, A]:
        raise NotImplementedError

    def getReward(self, oldState: State, action: Action, newState: State) -> float:
        raise NotImplementedError
