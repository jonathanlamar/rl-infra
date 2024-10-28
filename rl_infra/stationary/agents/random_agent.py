from random import randint

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context


class RandomAgent(StationaryBanditAgent[Policy]):
    def __init__(self, num_arms: int) -> None:
        self.policy = Policy()
        self.num_arms = num_arms

    def chooseAction(self, context: Context) -> Action:
        return randint(0, self.num_arms - 1)

    def updatePolicy(self, **kwargs) -> None: ...
