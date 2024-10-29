from typing import Protocol, TypeVar

from rl_infra.types.agent import Agent, Policy
from rl_infra.types.transition import Action, Context

P = TypeVar("P", bound=Policy)


class StationaryBanditAgent(Agent[Context, Action, P], Protocol[P]):
    policy: P
    numArms: int

    def __init__(self, numArms: int) -> None: ...
