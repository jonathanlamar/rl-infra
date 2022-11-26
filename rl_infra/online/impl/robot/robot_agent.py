from abc import ABC, abstractmethod
import random

from ...types.agent import Agent
from .robot_environment import RobotAction, RobotState


class RobotAgent(ABC, Agent[RobotState, RobotAction]):
    epsilon: float
    lastAction: RobotAction
    nextAction: RobotAction

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.lastAction = RobotAction.DO_NOTHING
        self.nextAction = RobotAction.MOVE_FORWARD

    @abstractmethod
    def choosePolicyAction(self, state: RobotState) -> RobotAction:
        ...

    def chooseRandomAction(self) -> RobotAction:
        return random.choice(
            [
                RobotAction.MOVE_FORWARD,
                RobotAction.MOVE_BACKWARD,
                RobotAction.TURN_LEFT,
                RobotAction.TURN_RIGHT,
                RobotAction.DO_NOTHING,
            ]
        )

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
