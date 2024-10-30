from typing import Protocol, TypeVar

from rl_infra.types.transition import Action, Context, Transition

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)


class Environment(Protocol[C, A]):
    r"""
    Generic interface for a bandit environment.  Contains a context (which is empty in
    the stationary setting) and a method for updating context based on agent action.
    This method returns the transition induced by the action, which contains the reward
    and updated context.
    """

    currentContext: C

    def update(self, action: A) -> Transition[C, A]: ...
