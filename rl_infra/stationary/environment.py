from numpy.random import normal

from rl_infra.types.environment import Environment
from rl_infra.types.transition import Action, Context, Transition


class StationaryBanditEnvironment(Environment[Context, Action]):
    def __init__(self, num_arms: int) -> None:
        self.currentContext = Context()
        self.num_arms = num_arms
        self.means = normal(0, 1, num_arms).tolist()

    def _getOptimalAction(self) -> Action:
        # TODO:  How is this implemented in the paper?  Is this best per expectation, or
        # best observed?
        return -1

    def update(self, action: Action) -> Transition:
        if action not in range(self.num_arms):
            raise RuntimeError(f"Action {action} is out of range.")

        return Transition(
            context=self.currentContext,
            action=action,
            optimalAction=action,
            newContext=self.currentContext,
            reward=normal(self.means[action], 1),
        )
