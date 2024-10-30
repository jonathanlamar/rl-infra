from typing import Generic, Type, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.environment import StationaryBanditEnvironment
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.history import History
from rl_infra.types.transition import Action, Context

A = TypeVar("A", bound=StationaryBanditAgent)


class StationaryBanditProblem(BanditProblem[Context, Action, A], Generic[A]):
    r"""
    Stationary bandit problem.  It operates in the stationary setting, which is context
    free.  As such, its agent must be a subclass of StationaryBanditAgent.
    """

    agent: A
    environment: StationaryBanditEnvironment
    history: History[Context, Action]

    def __init__(self, agentClass: Type[A], numArms: int) -> None:
        self.agent = agentClass(numArms)
        self.environment = StationaryBanditEnvironment(numArms)  # pyright: ignore
        self.history = History[Context, Action]()

    def play(self, numRounds: int) -> None:
        for _ in range(numRounds):
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
