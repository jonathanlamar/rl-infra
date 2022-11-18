import random

from ..types.agent import Agent
from ..types.environment import StepOutcome
from .robot_environment import RobotAction, RobotActionSpace, RobotState


class RobotAgent(Agent[RobotState, RobotAction]):
    def chooseAction(self, state: RobotState) -> RobotAction:
        # TODO: Implement
        return random.choice(list(RobotActionSpace))

    def updatePolicy(
        self, state: RobotState, action: RobotAction, outcome: StepOutcome
    ) -> None:
        # TODO: Implement
        pass