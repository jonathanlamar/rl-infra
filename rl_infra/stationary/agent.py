from typing import Protocol, TypeVar

from rl_infra.types.agent import Agent, Policy
from rl_infra.types.transition import Action, Context

P = TypeVar("P", bound=Policy)


class StationaryBanditAgent(Agent[Context, Action, P], Protocol[P]):
    r"""
    A stationary bandit agent. It expects a fixed number of arms which are specified in
    the constructor.  It also ignores the context entirely, which should be an empty
    dataclass in the stationary setting.
    """

    policy: P
    numArms: int

    def __init__(self, numArms: int) -> None: ...
