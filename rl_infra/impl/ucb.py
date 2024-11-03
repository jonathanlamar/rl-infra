import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context, Transition


class StationaryUcbPolicy(Policy):
    actionValues: NDArray[np.float64]
    ucbValues: NDArray[np.float64]
    numSteps: NDArray[np.int64]
    overallNumSteps: int
    explorationFactor: float

    @field_validator("numSteps")
    @classmethod
    def numStepsShouldAllBeNonNegative(
        cls, val: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        if np.any(val < 0):
            raise ValueError("Elements of numSteps should all be nonnegative.")
        return val

    @field_validator("overallNumSteps")
    @classmethod
    def overallNumStepsShouldBeNonNegative(cls, val: int) -> int:
        if val < 0:
            raise ValueError("overallNumSteps should be nonnegative.")
        return val

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

    def update(self, action: Action, reward: float) -> None:
        self.overallNumSteps += 1
        self._updateActionValue(action, reward)
        self._recomputeUcbValues()

    def _updateActionValue(self, action: Action, reward: float) -> None:
        existingValue = self.actionValues[action]
        existingSampleSize = self.numSteps[action]

        self.actionValues[action] = existingValue + (reward - existingValue) / (
            existingSampleSize + 1
        )
        self.numSteps[action] += 1

        # TODO: Find better way to validate upon changing an element of a numpy array.
        _ = self.numStepsShouldAllBeNonNegative(self.numSteps)

    def _recomputeUcbValues(self) -> None:
        self.ucbValues = self.actionValues + self.explorationFactor * np.sqrt(
            np.nan_to_num(np.log(self.overallNumSteps) / self.numSteps, nan=np.inf)
        )


class StationaryUcbAgent(StationaryBanditAgent[StationaryUcbPolicy]):
    r"""
    A stationary bandit agent based on the UCB estimator:
        A_t = argmax(Q_t(a) + c\sqrt{\log(t)/N_t(a)}),
    where c > 0 is a parameter that controls the degree of exploration, t is the overall
    number of steps taken by the agent, Q_t(a) is the action value, and N_t(a) is the
    number of times action a has been taken.  See Sutton and Barto, Section 2.7.
    """

    def __init__(self, numArms: int, c: float) -> None:
        self.policy = StationaryUcbPolicy(
            actionValues=np.zeros(numArms),
            ucbValues=np.ones(numArms),
            numSteps=np.zeros(numArms, dtype=np.int64),
            overallNumSteps=0,
            explorationFactor=c,
        )
        self.numArms = numArms

    def chooseAction(self, context: Context) -> Action:
        return self.policy.getMaxValueIndex()

    def updatePolicy(self, **kwargs) -> None:
        transition: Transition = kwargs["transition"]
        self.policy.update(transition.action, transition.reward)
