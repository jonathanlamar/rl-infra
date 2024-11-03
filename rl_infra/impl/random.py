from random import randint

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context


class RandomAgent(StationaryBanditAgent[Policy]):
    r"""
    A random stationary bandit agent.  This agent chooses actions completely at random.
    It is intended for use in testing and as a pathological example.
    """

    def __init__(self, numArms: int) -> None:
        self.policy = Policy()
        self.numArms = numArms

    def chooseAction(self, context: Context) -> Action:
        return randint(0, self.numArms - 1)

    def updatePolicy(self, **kwargs) -> None: ...
