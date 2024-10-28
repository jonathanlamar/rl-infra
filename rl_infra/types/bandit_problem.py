from typing import Protocol, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.types.agent import Agent, Policy
from rl_infra.types.environment import Environment
from rl_infra.types.history import History
from rl_infra.types.transition import Action, Context

C = TypeVar("C", bound=Context, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)
P = TypeVar("P", bound=Policy)


class BanditProblem(Protocol[C, A, P]):
    agent: Agent[C, A, P]
    environment: Environment[C, A]
    history: History[C, A]

    def play(self, num_rounds: int) -> None: ...

    def getRewardsVector(self) -> NDArray[float64]: ...

    def getOptimalActionHitsVector(self) -> NDArray[bool]: ...
