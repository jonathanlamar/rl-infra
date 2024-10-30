from typing import Protocol, Sequence, TypeVar

from numpy import float64
from numpy.typing import NDArray

from rl_infra.types.agent import Agent
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.transition import Action, Context

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)
Ag = TypeVar("Ag", bound=Agent)


class TestBed(Protocol[C, A, Ag]):
    r"""
    Generic interface for a bandit testbed.  This is a list of bandit problems along
    with a method for making all of them play.  Also contains methods for retrieving
    vectors of average rewards per time step (averaged over the list of bandit problems)
    and percent optimal action (average number of bandits choosing the optimal action at
    each time step).
    """

    bandits: Sequence[BanditProblem[C, A, Ag]]

    def play(self, numRounds: int) -> None: ...

    def getAverageRewardsVector(self) -> NDArray[float64]: ...

    def getPercentOptimalActionVector(self) -> NDArray[float64]: ...
