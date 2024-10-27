import logging
from typing import Protocol, TypeVar

from rl_infra.policy import Policy
from rl_infra.transition import Action, State

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=State, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)
P = TypeVar("P", bound=Policy)


class Agent(Protocol[S, A, P]):
    policy: P

    def chooseAction(self, state: S) -> A: ...
