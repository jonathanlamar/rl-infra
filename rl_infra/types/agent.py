from typing import Protocol, TypeVar

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.transition import Action, Context


class Policy(SerializableDataClass):
    r"""
    Dataclass representing the policy of the bandit agent.  In theory, a policy could be
    an approximator such as a neural network, but for this project, all policies are
    tabular.
    """


C = TypeVar("C", bound=Context, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)
P = TypeVar("P", bound=Policy)


class Agent(Protocol[C, A, P]):
    r"""
    Generic agent interface.  Every agent has a function for choosing an action based on
    context (which is empty for stationary bandit problems) as well as a function for
    updating the policy.
    """

    policy: P

    def chooseAction(self, context: C) -> A: ...

    def updatePolicy(self, **kwargs) -> None: ...
