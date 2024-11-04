import numpy as np
from numpy.typing import NDArray

from rl_infra.nonstationary.transition import (
    FEATURE_VECTOR_DIMENSION,
    NUM_ACTIONS,
    NonstationaryContext,
)
from rl_infra.types.environment import Environment
from rl_infra.types.transition import Action, Transition


class NonStationaryBanditEnvironment(Environment[NonstationaryContext, Action]):
    r"""
    Represents a nonstationary bandit environment.  This implements the "disjoint
    linear" models problem described in Li et al 2012, section 3.1
    """

    def __init__(self, numUsers: int, maxNumActions: int) -> None:
        self.numUsers = numUsers
        self.maxNumActions = maxNumActions
        self.currentContext = NonstationaryContext.randomNonstationaryContext()
        self.coefficientVectors = np.random.normal(
            0, 1, size=(NUM_ACTIONS, FEATURE_VECTOR_DIMENSION)
        )

    def update(self, action: Action) -> Transition:
        if action not in self.currentContext.availableActions:
            raise RuntimeError(f"Action {action} is out of range for this context.")

        actionIndex = self.currentContext.availableActions.index(action)
        calculatedReward = self._calculateMeanRewards()[actionIndex]

        return Transition(
            context=self.currentContext,
            action=action,
            newContext=self.currentContext,
            reward=calculatedReward,
        )

    def _calculateMeanRewards(self) -> NDArray:
        return np.diagonal(
            np.matmul(
                self.coefficientVectors[:, self.currentContext.availableActions],
                self.currentContext.featureVectors.T,
            )
        )

    def getOptimalAction(self) -> Action:
        bestAction = self._calculateMeanRewards().argmax()
        return self.currentContext.availableActions[bestAction]
