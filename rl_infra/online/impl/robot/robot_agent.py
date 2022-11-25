from abc import ABC, abstractmethod
import random

from ...types.agent import Agent
from .robot_environment import RobotAction, RobotState, RobotTransition


class RobotAgent(ABC, Agent[RobotState, RobotAction]):
    lastAction: RobotAction
    nextAction: RobotAction

    def __init__(self) -> None:
        self.lastAction = RobotAction.DO_NOTHING
        self.nextAction = RobotAction.MOVE_FORWARD

    @abstractmethod
    def chooseAction(self, state: RobotState) -> RobotAction:
        ...

    def chooseExploreAction(self, state: RobotState) -> RobotAction:
        action = self.nextAction
        self.lastAction = action

        if state.distanceSensor >= 75:
            self.nextAction = RobotAction.MOVE_FORWARD
        elif self.lastAction != RobotAction.MOVE_BACKWARD:
            self.nextAction = RobotAction.MOVE_BACKWARD
        else:
            self.nextAction = random.choice(
                [RobotAction.TURN_LEFT, RobotAction.TURN_RIGHT]
            )

        return action

    @abstractmethod
    def updatePolicy(self, transition: RobotTransition) -> None:
        ...
