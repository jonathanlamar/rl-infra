from typing import Protocol, TypeVar

from rl_infra.transition import Action, Context, SerializableDataClass


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

    def chooseAction(self, state: C) -> A: ...
