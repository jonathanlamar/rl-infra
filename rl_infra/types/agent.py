from typing import Protocol, TypeVar

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.transition import Action, Context


class Policy(SerializableDataClass):
    r"""
    Dataclass representing the policy of the bandit agent.  This is just an array
    representing expected action value.
    """


C = TypeVar("C", bound=Context, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)
P = TypeVar("P", bound=Policy)


class Agent(Protocol[C, A, P]):
    policy: P

    def chooseAction(self, context: C) -> A: ...

    def updatePolicy(self, **kwargs) -> None: ...
