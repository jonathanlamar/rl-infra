from typing import Protocol

from rl_infra.online.types.environment import StepOutcome
from rl_infra.online.types.space import Action, ActionSpace, State, StateSpace


class Agent(Protocol):
    stateSpace: StateSpace
    ActionSpace: ActionSpace

    def chooseAction(self, state: State) -> Action:
        raise NotImplementedError

    def updatePolicy(self, state: State, action: Action, outcome: StepOutcome) -> None:
        raise NotImplementedError
