from typing import Protocol, TypeVar

from rl_infra.transition import Action, Context, Transition

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)


class Environment(Protocol[C, A]):
    currentState: C

    def step(self, action: A) -> Transition[C, A]: ...

    def getReward(self, oldState: C, action: A, newState: C) -> float: ...

    def startNewEpisode(self) -> None: ...
