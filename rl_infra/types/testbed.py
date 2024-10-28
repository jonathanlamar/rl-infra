from typing import Protocol, TypeVar

from numpy import float64
from numpy.typing import NDArray

from rl_infra.types.agent import Policy
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.transition import Action, Context

C = TypeVar("C", bound=Context, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)
P = TypeVar("P", bound=Policy)


class TestBed(Protocol[C, A, P]):
    bandits: list[BanditProblem[C, A, P]]

    def play(self, num_rounds: int) -> None: ...

    def getAverageRewardsVector(self) -> NDArray[float64]: ...

    def getPercentOptimalActionVector(self) -> NDArray[float64]: ...
