import random
from typing import Protocol, TypeVar

from torch.nn import Module

from rl_infra.types.offline.model_service import ModelDbKey
from rl_infra.types.online.transition import Action, State

S = TypeVar("S", bound=State, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)
M = TypeVar("M", bound=Module)


class Agent(Protocol[S, A, M]):
    epsilon: float
    numEpisodesPlayed: int
    numEpochsTrained: int
    dbKey: ModelDbKey
    policy: M

    def chooseAction(self, state: S) -> A:
        """Choose action in epsilon-greedy manner, according to policy"""
        if random.random() < self.epsilon:
            return self.chooseRandomAction()
        else:
            return self.choosePolicyAction(state)

    def choosePolicyAction(self, state: S) -> A:
        ...

    def chooseRandomAction(self) -> A:
        ...
