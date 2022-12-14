from typing import Protocol, TypeVar

from ..types.environment import Action, State, StepOutcome

S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)


class Agent(Protocol[S, A]):
    def chooseAction(self, state: S) -> A:
        raise NotImplementedError

    def updatePolicy(self, state: S, action: A, outcome: StepOutcome[S]) -> None:
        raise NotImplementedError
