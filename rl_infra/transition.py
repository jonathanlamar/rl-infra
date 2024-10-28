from abc import ABC
from typing import Generic, TypeVar

from rl_infra.base_types import SerializableDataClass


class Context(SerializableDataClass):
    r"""
    Context for contextual bandit problems.  This is empty for stationary bandits.
    """


Action = int

C = TypeVar("C", bound=Context, covariant=True)
A = TypeVar("A", bound=Action, covariant=True)


# This is really an interface, but I have to use ABC here because pydantic does not
# support mixing in with protocols.
class Transition(ABC, SerializableDataClass, Generic[C, A]):
    context: C
    action: A
    newContext: C
    reward: float
