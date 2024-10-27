import logging
import random
from typing import Protocol, TypeVar

from rl_infra.types.offline.model_service import ModelDbKey
from rl_infra.types.online.policy import Policy
from rl_infra.types.online.transition import Action, State

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=State, covariant=False, contravariant=True)
A = TypeVar("A", bound=Action, covariant=True, contravariant=False)
P = TypeVar("P", bound=Policy)


class Agent(Protocol[S, A, P]):
    epsilon: float
    numEpisodesPlayed: int
    numEpochsTrained: int
    dbKey: ModelDbKey
    policy: P

    def chooseAction(self, state: S) -> A:
        """Choose action in epsilon-greedy manner, according to policy"""
        logger.debug("Choosing action")
        if random.random() < self.epsilon:
            return self.chooseRandomAction()
        else:
            return self.choosePolicyAction(state)

    def choosePolicyAction(self, state: S) -> A: ...

    def chooseRandomAction(self) -> A: ...
