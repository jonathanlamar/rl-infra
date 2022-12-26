import random
from abc import ABC, abstractmethod

from numpy import pi

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

        if state.distanceSweep[90] >= 75:
            self.nextAction = RobotAction.MOVE_FORWARD
        elif self.lastAction != RobotAction.MOVE_BACKWARD:
            self.nextAction = RobotAction.MOVE_BACKWARD
        else:
            # TODO: This could be greatly optimized
            # int_0^{180}(pi/180)*(1/2)*r^2dx where x is measured in degrees
            integrand = (state.distanceSweep**2) * (pi / 360)
            rightQuadrantIntegral = integrand[:90].sum() / 90
            leftQuadrantIntegral = integrand[90:].sum() / 90

            if rightQuadrantIntegral > leftQuadrantIntegral:
                self.nextAction = RobotAction.TURN_RIGHT
            else:
                self.nextAction = RobotAction.TURN_LEFT

        return action
