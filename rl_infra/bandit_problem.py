from typing import Protocol, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.agents.base_agent import Agent, Policy
from rl_infra.environment import Environment
from rl_infra.history import History
from rl_infra.transition import Action, Context

C = TypeVar("C", bound=Context, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)
P = TypeVar("P", bound=Policy)


class BanditProblem(Protocol[C, A, P]):
    agent: Agent[C, A, P]
    environment: Environment[C, A]
    history: History[C, A]

    def play_once(self) -> None: ...

    def getRewardsVector(self) -> NDArray[float64]: ...

    def getOptimalActionHitsVector(self) -> NDArray[bool]: ...
