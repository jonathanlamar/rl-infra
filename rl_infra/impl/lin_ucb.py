import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from rl_infra.nonstationary.context import (
    FEATURE_VECTOR_DIMENSION,
    NUM_ACTIONS,
    NonstationaryContext,
)
from rl_infra.types.agent import Agent, Policy
from rl_infra.types.transition import Action, Transition


class LinUcbPolicy(Policy):
    # b, the "response vector" - aggregated observed rewards
    actionValues: NDArray[np.float64] = np.zeros(
        shape=(NUM_ACTIONS, FEATURE_VECTOR_DIMENSION)
    )
    # A, the "covariance matrix"
    estimatedCovarianceMatrix: NDArray[np.float64] = np.concat(
        [
            np.identity(FEATURE_VECTOR_DIMENSION).reshape(
                FEATURE_VECTOR_DIMENSION, FEATURE_VECTOR_DIMENSION, 1
            )
        ]
        * NUM_ACTIONS,
        axis=2,
    )
    # Theta, the estimated coefficients
    estimatedCoefficientMatrix: NDArray[np.float64] = np.zeros(
        shape=(FEATURE_VECTOR_DIMENSION, NUM_ACTIONS)
    )
    # p, the UCB values
    ucbValues: NDArray[np.float64] = np.zeros(shape=(NUM_ACTIONS,))
    # alpha, the exploration factor
    explorationFactor: float

    @field_validator("explorationFactor")
    @classmethod
    def explorationFactorShouldBePositive(cls, val: float) -> float:
        if val <= 0:
            raise ValueError("explorationFactor should be positive.")
        return val

    def getMaxValueIndex(self, context: NonstationaryContext) -> int:
        valsOfInterest = self.ucbValues[context.availableActions]
        maxValMask = valsOfInterest == valsOfInterest.max()
        indexes = np.where(maxValMask)[0]
        bestIndex = np.random.choice(indexes)
        return context.availableActions[bestIndex]

    def update(
        self, action: Action, context: NonstationaryContext, reward: float
    ) -> None:
        self._updateCoefficientMatrix(context)
        self._updateUcbValues(context)
        self._updateCovarianceMatrix(action, context)
        self._updateActionValues(action, context, reward)

    def _updateCoefficientMatrix(self, context: NonstationaryContext) -> None:
        for action in context.availableActions:
            self.estimatedCoefficientMatrix[:, action] = np.matmul(
                np.pow(self.estimatedCovarianceMatrix[:, :, action], -1),
                self.actionValues[action],
            )

    def _updateUcbValues(self, context: NonstationaryContext) -> None:
        for i, action in enumerate(context.availableActions):
            featVec = context.featureVectors[:, i]
            coeffs = self.estimatedCoefficientMatrix[:, action]
            covMat = self.estimatedCovarianceMatrix[:, :, action]
            estimatedMean = np.dot(coeffs.T, featVec)
            confIntervalSize = np.matmul(
                featVec.T, np.matmul(np.pow(covMat, -1), featVec)
            )
            ucbEstimate = np.nan_to_num(
                estimatedMean + self.explorationFactor * np.sqrt(confIntervalSize),
                nan=np.inf,
            )
            self.ucbValues[action] = ucbEstimate

    def _updateCovarianceMatrix(
        self, action: Action, context: NonstationaryContext
    ) -> None:
        actionIndex = context.availableActions.index(action)
        x = context.featureVectors[:, actionIndex]
        self.estimatedCovarianceMatrix[:, :, action] += np.matmul(x, x.T)

    def _updateActionValues(
        self, action: Action, context: NonstationaryContext, reward: float
    ) -> None:
        actionIndex = context.availableActions.index(action)
        x = context.featureVectors[:, actionIndex]
        self.actionValues[action] += reward * x


class LinUcbUcbAgent(Agent[NonstationaryContext, Action, LinUcbPolicy]):
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
        return self.policy.getMaxValueIndex(context)

    def updatePolicy(self, **kwargs) -> None:
        transition: Transition = kwargs["transition"]
        self.policy.update(transition.action, transition.context, transition.reward)
