from random import randint, random

from pydantic import field_validator

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context, Transition


class ActionValuePolicy(Policy):
    actionValues: list[float]
    numSteps: list[int]

    @field_validator("numSteps")
    @classmethod
    def numStepsShouldAllBeNonNegative(cls, val: list[int]) -> list[int]:
        if any([v < 0 for v in val]):
            raise ValueError("Elements of numSteps should all be nonnegative.")
        return val

    def getMaxValueIndex(self) -> int:
        return self.actionValues.index(max(self.actionValues))

    def update(self, action: Action, reward: float) -> None:
        existingValue = self.actionValues[action]
        existingSampleSize = self.numSteps[action]

        self.actionValues[action] = existingValue + (reward - existingValue) / (
            existingSampleSize + 1
        )
        self.numSteps[action] += 1

        # TODO: Find better way to validate upon changing an element of a list.
        _ = self.numStepsShouldAllBeNonNegative(self.numSteps)


class ActionValueAgent(StationaryBanditAgent[ActionValuePolicy]):
    r"""
    A stationary bandit agent based on the action-value estimator.
    This agent chooses actions based on the action value estimator:
        Q_t(a) = average reward when a is taken prior to t
    See Sutton and Barto, Section 2.2.
    """

    def __init__(self, numArms: int, epsilon: float) -> None:
        self.policy = ActionValuePolicy(
            actionValues=[0] * numArms, numSteps=[0] * numArms
        )
        self.numArms = numArms
        self.epsilon = epsilon

    def chooseAction(self, context: Context) -> Action:
        if random() < self.epsilon:
            return randint(0, self.numArms - 1)
        else:
            return self.policy.getMaxValueIndex()

    def updatePolicy(self, **kwargs) -> None:
        transition: Transition = kwargs["transition"]
        self.policy.update(transition.action, transition.reward)
