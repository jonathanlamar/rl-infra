from typing import Protocol, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.types.agent import Agent
from rl_infra.types.environment import Environment
from rl_infra.types.history import History
from rl_infra.types.transition import Action, Context

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)
Ag = TypeVar("Ag", bound=Agent)


class BanditProblem(Protocol[C, A, Ag]):
    r"""
    Generic interface for a bandit problem.  This is defined as an agent, environment
    pair as well as a vector of transitions representing the history of
    agent-environment interactions.  It has methods for "playing" a number of rounds.  A
    round of play is defined as an agent choosing an action, observing the reward, and
    updating its policy.  The environment also updates its context.  This interface also
    specifies methods for getting vectors of observed rewards and optimal action hits
    from the history
    """

    agent: Ag
    environment: Environment[C, A]
    history: History[C, A]

    def play(self, numRounds: int) -> None:
        for _ in range(numRounds):
            self._playOnce()

    def _playOnce(self) -> None:
        context = self.environment.currentContext
        action = self.agent.chooseAction(context)
        transition = self.environment.update(action)
        optimalAction = self.environment.getOptimalAction()
        self.history.update(transition, optimalAction)  # pyright: ignore
        self.agent.updatePolicy(transition=transition)

    def getRewardsVector(self) -> NDArray[float64]:
        return self.history.getRewardsVector()

    def getOptimalActionHitsVector(self) -> NDArray[bool]:
        return self.history.getOptimalActionHitsVector()
