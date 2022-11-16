from typing import Protocol

from rl_infra.online.types import Agent
from rl_infra.online.types.environment import StepOutcome
from rl_infra.online.types.space import Action, State
from rl_infra.online.utils import RobotClient


class RobotAgent(Agent, Protocol):
    robot_driver: RobotClient

    def __init__(self, robot_driver: RobotClient) -> None:
        self.robot_driver = robot_driver

    def chooseAction(self, state: State) -> Action:
        return super().chooseAction(state)

    def updatePolicy(self, state: State, action: Action, outcome: StepOutcome) -> None:
        return super().updatePolicy(state, action, outcome)
