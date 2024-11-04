import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from rl_infra.nonstationary.transition import (
    FEATURE_VECTOR_DIMENSION,
    NUM_ACTIONS,
    NonstationaryContext,
)
from rl_infra.types.agent import Agent, Policy
from rl_infra.types.transition import Action, Transition


class LinUcbPolicy(Policy):
    # b "response vector" - aggregated observed rewards
    actionValues: NDArray[np.float64] = np.zeros(
        shape=(NUM_ACTIONS, FEATURE_VECTOR_DIMENSION)
    )
    # A "covariance matrix"
    estimatedCovarianceMatrix: NDArray[np.float64] = np.concat(
        [np.identity(FEATURE_VECTOR_DIMENSION).reshape(-1, -1, 1)] * NUM_ACTIONS, axis=2
    )
    # Theta estimated coefficients
    estimatedCoefficientMatrix: NDArray[np.float64] = np.zeros(
        shape=(FEATURE_VECTOR_DIMENSION, NUM_ACTIONS)
    )
    # p UCB values
    ucbValues: NDArray[np.float64] = np.zeros(shape=(NUM_ACTIONS,))
    # alpha
    explorationFactor: float

    @field_validator("explorationFactor")
    @classmethod
    def explorationFactorShouldBePositive(cls, val: float) -> float:
        if val <= 0:
            raise ValueError("explorationFactor should be positive.")
        return val

    def getMaxValueIndex(self) -> int:
        maxValMask = self.ucbValues == self.ucbValues.max()
        indexes = np.where(maxValMask)[0]
        return np.random.choice(indexes)

    def update(
        self, action: Action, context: NonstationaryContext, reward: float
    ) -> None:
        self._updateCoefficientMatrix(context)
        self._updateUcbValues(context)
        self._updateCovarianceMatrix(action, context)
        self._updateActionValues(action, context, reward)

    def _updateCoefficientMatrix(self, context: NonstationaryContext) -> None:
        for a in context.availableActions:
            self.estimatedCoefficientMatrix[:, a] = np.matmul(
                np.pow(self.estimatedCovarianceMatrix[:, :, a], -1),
                self.actionValues[a],
            )

    def _updateUcbValues(self, context: NonstationaryContext) -> None:
        for a in context.availableActions:
            featVec = context.featureVectors[:, a]
            coeffs = self.estimatedCoefficientMatrix[:, a]
            covMat = self.estimatedCovarianceMatrix[:, :, a]
            estimatedMean = np.dot(coeffs, featVec)
            confIntervalSize = np.matmul(
                featVec.T, np.matmul(np.pow(covMat, -1), featVec)
            )
            self.ucbValues[a] = (
                estimatedMean + self.explorationFactor * confIntervalSize
            )

    def _updateCovarianceMatrix(
        self, action: Action, context: NonstationaryContext
    ) -> None:
        x = context.featureVectors[action]
        self.estimatedCovarianceMatrix[action] += np.dot(x, x)

    def _updateActionValues(
        self, action: Action, context: NonstationaryContext, reward: float
    ) -> None:
        x = context.featureVectors[action]
        self.actionValues[action] += reward * x


class StationaryUcbAgent(Agent[NonstationaryContext, Action, LinUcbPolicy]):
    r"""
    A stationary bandit agent based on the UCB estimator:
        A_t = argmax(Q_t(a) + c\sqrt{\log(t)/N_t(a)}),
    where c > 0 is a parameter that controls the degree of exploration, t is the overall
    number of steps taken by the agent, Q_t(a) is the action value, and N_t(a) is the
    number of times action a has been taken.  See Sutton and Barto, Section 2.7.
    """

    def __init__(self, alpha: float) -> None:
        self.policy = LinUcbPolicy(explorationFactor=alpha)

    def chooseAction(self, context: NonstationaryContext) -> Action:
        return self.policy.getMaxValueIndex()

    def updatePolicy(self, **kwargs) -> None:
        transition: Transition = kwargs["transition"]
        self.policy.update(transition.action, transition.reward)
