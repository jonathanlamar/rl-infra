import random

import numpy as np
from numpy.typing import NDArray

from rl_infra.nonstationary.context import (
    FEATURE_VECTOR_DIMENSION,
    NUM_ACTIONS,
    NonstationaryContext,
)
from rl_infra.types.environment import Environment
from rl_infra.types.transition import Action, Transition

NUM_USERS = 2


class NonstationaryBanditEnvironment(Environment[NonstationaryContext, Action]):
    r"""
    Represents a nonstationary bandit environment.  This implements the "disjoint
    linear" models problem described in Li et al 2012, section 3.1
    """

    def __init__(self) -> None:
        self.contexts = [
            NonstationaryContext.randomNonstationaryContext(user=i)
            for i in range(NUM_USERS)
        ]
        self.currentContext = random.choice(self.contexts)
        self.coefficientVectors = np.random.normal(
            0, 1, size=(FEATURE_VECTOR_DIMENSION, NUM_ACTIONS)
        )

    def update(self, action: Action) -> Transition:
        if action not in self.currentContext.availableActions:
            raise RuntimeError(f"Action {action} is out of range for this context.")

        actionIndex = self.currentContext.availableActions.index(action)
        calculatedReward = self._calculateMeanRewards()[actionIndex]
        oldContext = self.currentContext

        return Transition(
            context=oldContext,
            action=action,
            newContext=self.currentContext,
            reward=calculatedReward,
        )

    def _calculateMeanRewards(self) -> NDArray:
        return np.diagonal(
            np.matmul(
                self.coefficientVectors[:, self.currentContext.availableActions].T,
                self.currentContext.featureVectors,
            )
        )

    def getOptimalAction(self) -> Action:
        bestAction = self._calculateMeanRewards().argmax()
        return self.currentContext.availableActions[bestAction]
