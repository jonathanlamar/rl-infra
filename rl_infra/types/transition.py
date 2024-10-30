from abc import ABC
from typing import Generic, TypeVar

from rl_infra.types.base_types import SerializableDataClass


class Context(SerializableDataClass):
    r"""
    Context for contextual bandit problems.  This is empty for stationary bandits.
    """


Action = int

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)


class Transition(ABC, SerializableDataClass, Generic[C, A]):
    r"""
    Dataclass representing a generic agent-environment interaction in a bandit problem.
    """

    context: C
    action: A
    optimalAction: A
    newContext: C
    reward: float
