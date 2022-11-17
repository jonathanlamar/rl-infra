from dataclasses import dataclass
from typing import Protocol, TypeVar


# States will vary quite a bit between implementations, so I am just using this
# class as a type stub.
class State:
    pass


# Action spaces should always be countable.  Maybe we can change this in the
# future, but I will be using int-typed enums for action spaces.
Action = int
S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=True)


@dataclass
class StepOutcome(Protocol[S]):
    newState: S
    reward: float
    isTerminal: bool


class Environment(Protocol[S, A]):
    currentState: S

    def step(self, action: A) -> StepOutcome[S]:
        raise NotImplementedError
