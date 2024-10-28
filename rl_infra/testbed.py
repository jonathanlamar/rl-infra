from typing import Protocol, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.agents.base_agent import Policy
from rl_infra.bandit_problem import BanditProblem
from rl_infra.transition import Action, Context

C = TypeVar("C", bound=Context, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)
P = TypeVar("P", bound=Policy)


class TestBed(Protocol[C, A, P]):
    bandits: list[BanditProblem[C, A, P]]

    def play(self, num_rounds: int) -> None: ...

    def get_avg_reward_plot(self) -> NDArray[float64]: ...

    def get_optimal_action_plot(self) -> NDArray[bool]: ...
