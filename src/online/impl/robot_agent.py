from typing import Protocol

from online.types import Agent
from online.types.environment import StepOutcome
from online.types.space import Action, State
from online.utils import RobotDriver


class RobotAgent(Agent, Protocol):
    robot_driver: RobotDriver

    def __init__(self, robot_driver: RobotDriver) -> None:
        self.robot_driver = robot_driver

    def chooseAction(self, state: State) -> Action:
        return super().chooseAction(state)

    def updatePolicy(self, state: State, action: Action, outcome: StepOutcome) -> None:
        return super().updatePolicy(state, action, outcome)
