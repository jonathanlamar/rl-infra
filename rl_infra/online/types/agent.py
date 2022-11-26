import random
from typing import Protocol, TypeVar

from ..types.environment import Action, State

S = TypeVar("S", bound=State, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)


class Agent(Protocol[S, A]):
    epsilon: float

    def chooseAction(self, state: S) -> A:
        """Choose action in epsilon-greedy manner, according to policy"""
        if random.random() < self.epsilon:
            return self.chooseRandomAction()
        else:
            return self.choosePolicyAction(state)

    def choosePolicyAction(self, state: S) -> A:
        raise NotImplementedError

    def chooseRandomAction(self) -> A:
        raise NotImplementedError
