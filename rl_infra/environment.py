from typing import Protocol, TypeVar

from rl_infra.transition import Action, State, Transition

S_co = TypeVar("S_co", bound=State, covariant=True)
A_co = TypeVar("A_co", bound=Action, covariant=True)


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class Environment(Protocol[S, A]):
    currentState: S

    def step(self, action: A) -> Transition[S, A]: ...

    def getReward(self, oldState: S, action: A, newState: S) -> float: ...

    def startNewEpisode(self) -> None: ...
