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
    bandits: Sequence[BanditProblem[C, A, Ag]]

    def play(self, num_rounds: int) -> None: ...

    def getAverageRewardsVector(self) -> NDArray[float64]: ...

    def getPercentOptimalActionVector(self) -> NDArray[float64]: ...
