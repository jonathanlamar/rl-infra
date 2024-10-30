from numpy.random import normal

from rl_infra.types.environment import Environment
from rl_infra.types.transition import Action, Context, Transition


class StationaryBanditEnvironment(Environment[Context, Action]):
    r"""
    Represents a stationary bandit environment.  Contains an empty context and a no-op
    method for updating context based on agent action, which returns the reward in a
    Transition instance.
    """

    def __init__(self, numArms: int) -> None:
        self.currentContext = Context()
        self.num_arms = numArms
        self.means = normal(0, 1, numArms).tolist()
        self._optimalAction = self.means.index(max(self.means))

    def update(self, action: Action) -> Transition:
        if action not in range(self.num_arms):
            raise RuntimeError(f"Action {action} is out of range.")

        return Transition(
            context=self.currentContext,
            action=action,
            optimalAction=self._optimalAction,
            newContext=self.currentContext,
            reward=normal(self.means[action], 1),
        )
