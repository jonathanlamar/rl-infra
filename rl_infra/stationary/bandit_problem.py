from typing import Protocol, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.environment import StationaryBanditEnvironment
from rl_infra.types.agent import Policy
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.history import History
from rl_infra.types.transition import Action, Context

P = TypeVar("P", bound=Policy)


class StationaryBanditProblem(BanditProblem[Context, Action, P], Protocol[P]):
    agent: StationaryBanditAgent
    environment: StationaryBanditEnvironment
    history: History[Context, Action]

    def play(self, num_rounds: int) -> None:
        for _ in range(num_rounds):
            self._playOnce()

    def _playOnce(self) -> None:
        context = self.environment.currentContext
        action = self.agent.chooseAction(context)
        transition = self.environment.update(action)
        self.history.update(transition)
        self.agent.updatePolicy(transition=transition)

    def getRewardsVector(self) -> NDArray[float64]:
        return self.history.getRewardsVector()

    def getOptimalActionHitsVector(self) -> NDArray[bool]:
        return self.history.getOptimalActionHitsVector()
