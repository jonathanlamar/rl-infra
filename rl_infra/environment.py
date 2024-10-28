from typing import Protocol, TypeVar

from rl_infra.transition import Action, Context, Transition

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)


class Environment(Protocol[C, A]):
    currentContext: C

    def update(self, action: A) -> Transition[C, A]: ...
